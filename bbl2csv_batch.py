import sys
import os
import csv
import logging
import argparse
from struct import pack, unpack
from collections import namedtuple
from enum import Enum, IntEnum
from typing import BinaryIO, Dict, Iterator, List, Optional, Tuple, Union, Callable, Any

# ==========================================
# PART 1: Types & Tools
# ==========================================

Number = Union[int, float]

class FrameType(Enum):
    INTER = 'P'
    INTRA = 'I'
    GPS = 'G'
    SLOW = 'S'
    GPS_HOME = 'H'
    EVENT = 'E'

class EventType(IntEnum):
    SYNC_BEEP = 0
    AUTOTUNE_CYCLE_START = 10
    AUTOTUNE_CYCLE_RESULT = 11
    AUTOTUNE_TARGETS = 12
    INFLIGHT_ADJUSTMENT = 13
    LOGGING_RESUME = 14
    GTUNE_CYCLE_RESULT = 20
    FLIGHT_MODE = 30
    TWITCH_TEST = 40
    CUSTOM = 250
    CUSTOM_BLANK = 251
    LOG_END = 255

# Forward declarations
DecodedValue = Union[int, Tuple]
Decoder = Callable[[Iterator[int], Optional["Context"]], DecodedValue]
Predictor = Callable[[int, "Context"], int]

class FieldDef:
    def __init__(self, frame_type: FrameType, name: Optional[str] = None,
                 signed: Optional[int] = None, predictor: Optional[int] = None,
                 encoding: Optional[int] = None, decoderfun: Optional[Decoder] = None,
                 predictorfun: Optional[Predictor] = None):
        self.type = frame_type
        self.name = name
        self.signed = signed
        self.predictor = predictor
        self.encoding = encoding
        self.decoderfun = decoderfun
        self.predictorfun = predictorfun

FieldDefs = Dict[FrameType, List[FieldDef]]
Headers = Dict[str, Union[str, Number, List[Number]]]

Frame = namedtuple('Frame', 'type data')
Event = namedtuple('Event', 'type data')
EventParser = Callable[[Iterator[int]], Optional[dict]]

# --- Tools ---
def map_to(key: Any, amap: dict) -> Callable:
    def decorator(fun: Callable) -> Callable:
        amap[key] = fun
        return fun
    return decorator

def toint32(word):
    return unpack('i', pack('I', word))[0]

def sign_extend_24bit(bits): return toint32(bits | 0xFF000000) if bits & 0x800000 else bits
def sign_extend_16bit(word): return toint32(word | 0xFFFF0000) if word & 0x8000 else word
def sign_extend_14bit(word): return toint32(word | 0xFFFFC000) if word & 0x2000 else word
def sign_extend_8bit(byte):  return toint32(byte | 0xFFFFFF00) if byte & 0x80 else byte
def sign_extend_6bit(byte):  return toint32(byte | 0xFFFFFFC0) if byte & 0x20 else byte
def sign_extend_4bit(nibble): return toint32(nibble | 0xFFFFFFF0) if nibble & 0x08 else nibble
def sign_extend_2bit(byte):  return toint32(byte | 0xFFFFFFFC) if byte & 0x02 else byte

def _trycast(s: str) -> Union[Number, str]:
    if s.startswith("0x"): return int(s, 16)
    try: return int(s)
    except ValueError:
        try: return float(s)
        except ValueError: return s

def _is_ascii(s: bytes) -> bool:
    try: s.decode('ascii'); return True
    except UnicodeDecodeError: return False

# ==========================================
# PART 2: Errors & Defaults
# ==========================================

class InvalidHeaderException(Exception):
    def __init__(self, data: bytes, position: int):
        super().__init__(f"Invalid header at 0x{position:X}: {data}")

class HeaderDefaults:
    defaults = {
        "Data version": 1, "I interval": 1, "P interval": 0,
        "minthrottle": 0, "motorOutput": [0, 0], "vbatref": 0,
    }
    @classmethod
    def inspect(cls, headers: Headers):
        pass 
    @classmethod 
    @property
    def data_version(cls) -> int: return cls.defaults["Data version"]
    @classmethod
    @property
    def i_interval(cls) -> int: return cls.defaults["I interval"]
    @classmethod
    @property
    def p_interval(cls) -> int: return cls.defaults["P interval"]
    @classmethod
    @property
    def minthrottle(cls) -> int: return cls.defaults["minthrottle"]
    @classmethod
    @property
    def motor_output(cls) -> int: return cls.defaults["motorOutput"]
    @classmethod
    @property
    def vbatref(cls) -> int: return cls.defaults["vbatref"]

# ==========================================
# PART 3: Context
# ==========================================

class Context:
    def __init__(self, headers: Headers, field_defs: FieldDefs):
        self.headers = headers
        self.data_version = headers.get("Data version", HeaderDefaults.data_version)
        self.field_defs = field_defs
        self.field_def_counts = {k: len(v) for k, v in field_defs.items()}
        self.frame_count = 0
        self.frame_type = None
        self.field_index = 0
        self.past_frames = (Frame(FrameType.INTRA, b''), Frame(FrameType.INTRA, b''), Frame(FrameType.INTRA, b''))
        self.last_gps_frame = Frame(FrameType.GPS, b'')
        self.last_gps_home_frame = Frame(FrameType.GPS_HOME, b'')
        self.current_frame = tuple()
        self.last_iter = -1
        self._names_to_indices = dict()
        for ftype in FrameType:
            if ftype in self.field_defs:
                self._names_to_indices[ftype] = dict()
                for i, fdef in enumerate(self.field_defs[ftype]):
                    self._names_to_indices[ftype][fdef.name] = i
        self.read_frame_count = 0
        self.invalid_frame_count = 0
        self.i_interval = self.headers.get("I interval", HeaderDefaults.i_interval)
        self.skipped_frames = 0
        if self.i_interval < 1: self.i_interval = 1
        p_interval = self.headers.get("P interval", HeaderDefaults.p_interval)
        if isinstance(p_interval, int):
            self.p_interval_num = 1
            self.p_interval_denom = p_interval
        else:
            num, denom = p_interval.split('/')
            self.p_interval_num = int(num)
            self.p_interval_denom = int(denom)

    def add_frame(self, frame: Frame):
        if frame.type == FrameType.INTRA:
            self.past_frames = (frame, frame, frame)
        elif frame.type == FrameType.GPS:
            self.last_gps_frame = frame
        elif frame.type == FrameType.GPS_HOME:
            self.last_gps_home_frame = frame
        else:
            self.past_frames = (frame, self.past_frames[0], self.past_frames[1])
        self.frame_count += 1

    def get_past_value(self, age: int, default: Number = 0) -> Number:
        try: return self.past_frames[age].data[self.field_index]
        except (KeyError, IndexError): return default

    def get_current_value_by_name(self, frame_type: FrameType, field_name: str, default: Number = 0) -> Number:
        try: return self.current_frame[self._names_to_indices[frame_type][field_name]]
        except (KeyError, IndexError): return default

    def should_have_frame_at(self, index: int) -> bool:
        return (index % self.i_interval + self.p_interval_num - 1) % self.p_interval_denom < self.p_interval_num

    def count_skipped_frames(self) -> int:
        if self.last_iter == -1: return 0
        index = self.last_iter + 1
        while not self.should_have_frame_at(index): index += 1
        return index - self.last_iter - 1

    @property
    def stats(self) -> dict:
        skipped = self.read_frame_count - self.frame_count - self.invalid_frame_count
        return {
            "total": self.read_frame_count, "parsed": self.frame_count, "skipped": skipped,
            "invalid": self.invalid_frame_count,
        }

# ==========================================
# PART 4: Decoders & Predictors
# ==========================================

decoder_map = dict()
predictor_map = dict()

# --- Decoders ---
@map_to(0, decoder_map)
def _signed_vb(data: Iterator[int], ctx: Optional[Context] = None) -> DecodedValue:
    value = _unsigned_vb(data, ctx)
    value = ((value % 0x100000000) >> 1) ^ -(value & 1)
    return value

@map_to(1, decoder_map)
def _unsigned_vb(data: Iterator[int], ctx: Optional[Context] = None) -> DecodedValue:
    shift, result = 0, 0
    for i in range(5):
        try:
            byte = next(data)
        except StopIteration:
            return 0
        result = result | ((byte & ~0x80) << shift)
        if byte < 128: return result
        shift += 7
    return 0

@map_to(3, decoder_map)
def _neg_14bit(data: Iterator[int], ctx: Optional[Context] = None) -> DecodedValue:
    return -sign_extend_14bit(_unsigned_vb(data, ctx))

@map_to(6, decoder_map)
def _tag8_8svb(data: Iterator[int], ctx: Optional[Context] = None) -> DecodedValue:
    group_count = 8
    fdeflen = ctx.field_def_counts[ctx.frame_type]
    for i in range(ctx.field_index + 1, ctx.field_index + 8):
        if i == fdeflen:
            group_count = (fdeflen-1) - ctx.field_index
            break
        if ctx.field_defs[ctx.frame_type][i].encoding != 6:
            group_count = i - ctx.field_index
            break
    if group_count == 1:
        return _signed_vb(data, ctx)
    else:
        header = next(data)
        values = ()
        for _ in range(group_count):
            values += (_signed_vb(data, ctx) if header & 0x01 else 0,)
            header >>= 1
        return values

@map_to(7, decoder_map)
def _tag2_3s32(data: Iterator[int], ctx: Optional[Context] = None) -> DecodedValue:
    lead = next(data)
    shifted = lead >> 6
    if shifted == 0:
        return (sign_extend_2bit((lead >> 4) & 0x03), sign_extend_2bit((lead >> 2) & 0x03), sign_extend_2bit(lead & 0x03))
    elif shifted == 1:
        lead2 = next(data)
        return (sign_extend_4bit(lead & 0x0F), sign_extend_4bit(lead2 >> 4), sign_extend_4bit(lead2 & 0x0F))
    elif shifted == 2:
        lead2, lead3 = next(data), next(data)
        return (sign_extend_6bit(lead & 0x3F), sign_extend_6bit(lead2 & 0x3F), sign_extend_6bit(lead3 & 0x3F))
    elif shifted == 3:
        values = ()
        for _ in range(3):
            field_type = lead & 0x03
            if field_type == 0: values += (sign_extend_8bit(next(data)),)
            elif field_type == 1: 
                v1, v2 = next(data), next(data)
                values += (sign_extend_16bit(v1 | (v2 << 8)),)
            elif field_type == 2:
                v1, v2, v3 = next(data), next(data), next(data)
                values += (sign_extend_24bit(v1 | (v2 << 8) | (v3 << 16)),)
            elif field_type == 3:
                v1, v2, v3, v4 = next(data), next(data), next(data), next(data)
                values += (v1 | (v2 << 8) | (v3 << 16) | (v4 << 24),)
            lead >>= 2
        return values
    return 0, 0, 0

@map_to(8, decoder_map)
def _tag8_4s16_versioned(data_version: int) -> Decoder:
    return _tag8_4s16_v2

def _tag8_4s16_v2(data: Iterator[int], ctx: Optional[Context] = None) -> DecodedValue:
    selector = next(data)
    values = ()
    nibble_index = 0
    buffer = 0
    for _ in range(4):
        field_type = selector & 0x03
        if field_type == 0: values += (0,)
        elif field_type == 1:
            if nibble_index == 0:
                buffer = next(data)
                values += (sign_extend_4bit(buffer >> 4),)
                nibble_index = 1
            else:
                values += (sign_extend_4bit(buffer & 0x0F),)
                nibble_index = 0
        elif field_type == 2:
            if nibble_index == 0:
                values += (sign_extend_8bit(next(data)),)
            else:
                v1 = (buffer & 0x0F) << 4
                buffer = next(data)
                v1 |= buffer >> 4
                values += (sign_extend_8bit(v1),)
        elif field_type == 3:
            if nibble_index == 0:
                v1, v2 = next(data), next(data)
                values += (sign_extend_16bit((v1 << 8) | v2),)
            else:
                v1, v2 = next(data), next(data)
                values += (sign_extend_16bit(((buffer & 0x0F) << 12) | (v1 << 4) | (v2 >> 4)),)
                buffer = v2
        selector >>= 2
    return values

@map_to(9, decoder_map)
def _null(data: Iterator[int], ctx: Optional[Context] = None) -> DecodedValue: return 0
@map_to(10, decoder_map)
def _tag2_3svariable(_: Iterator[int], __: Optional[Context] = None) -> DecodedValue: return 0

# --- Predictors ---
@map_to(0, predictor_map)
def _noop(new: Number, _: Context) -> Number: return new
@map_to(1, predictor_map)
def _previous(new: Number, ctx: Context) -> Number: return new + ctx.get_past_value(0, 0)
@map_to(2, predictor_map)
def _straight_line(new: Number, ctx: Context) -> Number:
    prev, prev2 = ctx.get_past_value(0), ctx.get_past_value(1, ctx.get_past_value(0))
    return new + 2 * prev - prev2
@map_to(3, predictor_map)
def _average2(new: Number, ctx: Context) -> Number:
    prev, prev2 = ctx.get_past_value(0), ctx.get_past_value(1, ctx.get_past_value(0))
    return new + int((prev + prev2) / 2)
@map_to(4, predictor_map)
def _minthrottle(new: Number, ctx: Context) -> Number: return new + ctx.headers.get("minthrottle", 0)
@map_to(5, predictor_map)
def _motor0(new: Number, ctx: Context) -> Number: return new + ctx.get_current_value_by_name(FrameType.INTRA, "motor[0]")
@map_to(6, predictor_map)
def _increment(_: Number, ctx: Context) -> Number: return 1 + ctx.get_past_value(0) + ctx.count_skipped_frames()
@map_to(7, predictor_map)
def _home_coord_0(new: Number, ctx: Context) -> Number: return new + (ctx.last_gps_home_frame.data[0] if ctx.last_gps_home_frame.data else 0)
@map_to(256, predictor_map)
def _home_coord_1(new: Number, ctx: Context) -> Number: return new + (ctx.last_gps_home_frame.data[1] if ctx.last_gps_home_frame.data else 0)
@map_to(8, predictor_map)
def _1500(new: Number, _: Context) -> Number: return new + 1500
@map_to(9, predictor_map)
def _vbatref(new: Number, ctx: Context) -> Number: return new + ctx.headers.get("vbatref", 0)
@map_to(10, predictor_map)
def _last_main_frame_time(new: Number, ctx: Context) -> Number: return new + ctx.get_past_value(1, 0)
@map_to(11, predictor_map)
def _minmotor(new: Number, ctx: Context) -> Number: return new + ctx.headers.get("motorOutput", [0])[0]

# ==========================================
# PART 5: Events
# ==========================================

event_map = dict()
@map_to(EventType.SYNC_BEEP, event_map)
def sync_beep(data: Iterator[int]) -> Optional[dict]: return {"time": _unsigned_vb(data)}
@map_to(EventType.FLIGHT_MODE, event_map)
def flight_mode(data: Iterator[int]) -> Optional[dict]: return {"new_flags": _unsigned_vb(data), "old_flags": _unsigned_vb(data)}
@map_to(EventType.LOG_END, event_map)
def logging_end(data: Iterator[int]) -> Optional[dict]: return None
@map_to(EventType.AUTOTUNE_TARGETS, event_map)
def autotune_targets(_): pass
@map_to(EventType.AUTOTUNE_CYCLE_START, event_map)
def autotune_cycle_start(_): pass
@map_to(EventType.AUTOTUNE_CYCLE_RESULT, event_map)
def autotune_cycle_result(_): pass
@map_to(EventType.GTUNE_CYCLE_RESULT, event_map)
def gtune_cycle_result(_): pass
@map_to(EventType.CUSTOM_BLANK, event_map)
def custom_blank(_): pass
@map_to(EventType.TWITCH_TEST, event_map)
def twitch_test(_): pass
@map_to(EventType.INFLIGHT_ADJUSTMENT, event_map)
def inflight_adjustment(_): pass
@map_to(EventType.LOGGING_RESUME, event_map)
def logging_resume(_): pass


# ==========================================
# PART 6: Reader
# ==========================================

class Reader:
    def __init__(self, path: str, log_index: Optional[int] = None, allow_invalid_header: bool = False):
        self._headers = {}
        self._field_defs = {}
        self._log_index = 0
        self._header_size = 0
        self._path = path
        self._frame_data_ptr = 0
        self._log_pointers = []
        self._frame_data = b''
        self._frame_data_len = 0
        self._allow_invalid_header = allow_invalid_header
        
        with open(path, "rb") as f:
            if not f.seekable(): raise IOError("Input file must be seekable")
            self._find_pointers(f)
        if log_index is not None:
            self.set_log_index(log_index)

    def set_log_index(self, index: int):
        if index == self._log_index: return
        if index < 1 or self.log_count < index: raise RuntimeError(f"Invalid log_index: {index}")
        start = self._log_pointers[index - 1]
        with open(self._path, "rb") as f:
            f.seek(start)
            self._update_headers(f)
            f.seek(start + self._header_size)
            size = self._log_pointers[index] - start - self._header_size if index < self.log_count else None
            self._frame_data = f.read(size) if size is not None else f.read()
        self._log_index = index
        self._frame_data_ptr = 0
        self._frame_data_len = len(self._frame_data)
        self._build_field_defs()

    def _update_headers(self, f: BinaryIO):
        start = f.tell()
        while True:
            line = self._read_header_line(f)
            if not self._parse_header_line(line): break
        self._header_size = f.tell() - start

    def _read_header_line(self, f: BinaryIO) -> Optional[bytes]:
        result = bytes()
        while True:
            byte = f.read(1)
            if not byte: return result
            elif byte == b'I' and len(result) == 0:
                f.seek(-1, 1); return None
            elif byte == b'\n': return result + b'\n'
            elif not _is_ascii(byte) and result.startswith(b'H'):
                if self._allow_invalid_header:
                    f.seek(-(len(result) - result.find(b'I') + 1), 1); return None
                else: raise InvalidHeaderException(result, f.tell())
            result += byte

    def _parse_header_line(self, data: Optional[bytes]) -> bool:
        if not data or data[0] != 72: return False
        line = data.decode().replace("H ", "", 1)
        try: name, value = line.split(':', 1)
        except ValueError: return False
        self._headers[name.strip()] = [_trycast(s.strip()) for s in value.split(',')] if ',' in value else _trycast(value.strip())
        return True

    def _find_pointers(self, f: BinaryIO):
        start = f.tell()
        first_line = f.readline()
        f.seek(start)
        content = f.read()
        new_index = content.find(first_line)
        step = len(first_line)
        while -1 < new_index:
            self._log_pointers.append(new_index)
            new_index = content.find(first_line, new_index + step + 1)

    def _build_field_defs(self):
        for frame_type in FrameType:
            for header_key, header_value in self._headers.items():
                if "Field " + frame_type.value not in header_key: continue
                if frame_type not in self._field_defs:
                    self._field_defs[frame_type] = [FieldDef(frame_type) for _ in range(len(header_value))]
                prop = header_key.split(" ", 2)[-1]
                for i, framedef_value in enumerate(header_value):
                    if prop == "predictor": self._field_defs[frame_type][i].predictorfun = predictor_map[framedef_value]
                    elif prop == "encoding":
                        decoder = decoder_map[framedef_value]
                        if decoder.__name__.endswith("_versioned"): decoder = decoder(self._headers.get("Data version", 1))
                        self._field_defs[frame_type][i].decoderfun = decoder
                    elif prop == "name": self._field_defs[frame_type][i].name = framedef_value
                    else: setattr(self._field_defs[frame_type][i], prop, framedef_value)
        if FrameType.INTER in self._field_defs:
            for i, fdef in enumerate(self._field_defs[FrameType.INTER]):
                fdef.name = self._field_defs[FrameType.INTRA][i].name

    @property
    def log_count(self) -> int: return len(self._log_pointers)
    @property
    def headers(self) -> Headers: return dict(self._headers)
    @property
    def field_defs(self) -> Dict[FrameType, List[FieldDef]]: return dict(self._field_defs)
    def value(self) -> int: return self._frame_data[self._frame_data_ptr]
    def has_subsequent(self, data: bytes) -> bool: return self._frame_data[self._frame_data_ptr:self._frame_data_ptr + len(data)] == data
    def tell(self) -> int: return self._frame_data_ptr
    def seek(self, n: int): self._frame_data_ptr = n
    def __iter__(self) -> Iterator[Optional[int]]: return self
    def __next__(self) -> int:
        if self._frame_data_len == self._frame_data_ptr: raise StopIteration
        byte = self._frame_data[self._frame_data_ptr]
        self._frame_data_ptr += 1
        return byte

# ==========================================
# PART 7: Parser
# ==========================================

class Parser:
    def __init__(self, reader: Reader):
        self._reader = reader
        self._events = []
        self._headers = {}
        self._field_names = []
        self._end_of_log = False
        self._ctx = None
        self.set_log_index(reader._log_index if reader._log_index else 1)

    def set_log_index(self, index: int):
        self._events = []
        self._end_of_log = False
        self._reader.set_log_index(index)
        self._headers = {k: v for k, v in self._reader.headers.items() if "Field" not in k}
        self._ctx = Context(self._headers, self._reader.field_defs)
        self._field_names = []
        for ftype in [FrameType.INTRA, FrameType.SLOW, FrameType.GPS]:
            if ftype in self._reader.field_defs:
                self._field_names += filter(lambda x: x is not None and x not in self._field_names,
                                            map(lambda x: x.name, self._reader.field_defs[ftype]))

    @staticmethod
    def load(path: str, log_index: int = 1) -> "Parser":
        return Parser(Reader(path, log_index))

    def frames(self) -> Iterator[Frame]:
        field_defs = self._reader.field_defs
        last_slow = None
        last_gps = None
        ctx = self._ctx
        reader = self._reader
        
        for byte in reader:
            try: ftype = FrameType(chr(byte))
            except ValueError: continue 

            ctx.frame_type = ftype
            
            if ftype == FrameType.EVENT:
                if not self._parse_event_frame(reader): ctx.invalid_frame_count += 1
                ctx.read_frame_count += 1
                if self._end_of_log: break
                continue

            if ftype not in field_defs:
                ctx.invalid_frame_count += 1
                ctx.read_frame_count += 1
                continue

            frame = self._parse_frame(field_defs[ftype], reader)

            if ftype == FrameType.SLOW:
                last_slow = frame
                ctx.read_frame_count += 1
                continue
            elif ftype == FrameType.GPS:
                last_gps = frame
                ctx.read_frame_count += 1
                continue
            elif ftype == FrameType.GPS_HOME:
                ctx.add_frame(frame)
                ctx.read_frame_count += 1
                continue

            ctx.read_frame_count += 1
            
            # Merging logic similar to Rust
            extra_data = []
            if FrameType.SLOW in field_defs:
                extra_data += last_slow.data if last_slow else [""] * len(field_defs[FrameType.SLOW])
            if FrameType.GPS in field_defs:
                extra_data += list(last_gps.data[1:]) if last_gps else [""] * (len(field_defs[FrameType.GPS]) - 1)

            frame = Frame(ftype, frame.data + tuple(extra_data))
            ctx.add_frame(frame)
            yield frame

    def _parse_frame(self, fdefs: List[FieldDef], reader: Reader) -> Frame:
        result = ()
        ctx = self._ctx
        ctx.field_index = 0
        field_count = ctx.field_def_counts[ctx.frame_type]
        while ctx.field_index < field_count:
            ctx.current_frame = result
            fdef = fdefs[ctx.field_index]
            rawvalue = fdef.decoderfun(reader, ctx)
            if isinstance(rawvalue, tuple):
                value = ()
                for v in rawvalue:
                    fdef = fdefs[ctx.field_index]
                    value += (fdef.predictorfun(v, ctx),)
                    ctx.field_index += 1
                result += value
            else:
                value = fdef.predictorfun(rawvalue, ctx)
                ctx.field_index += 1
                result += (value,)
        return Frame(ctx.frame_type, result)

    def _parse_event_frame(self, reader: Reader) -> bool:
        try:
            byte = next(reader)
            event_type = EventType(byte)
        except (ValueError, StopIteration): return False
        
        parser_fun = event_map.get(event_type)
        if parser_fun:
            event_data = parser_fun(reader)
            self._events.append(Event(event_type, event_data))
        if event_type == EventType.LOG_END:
            self._end_of_log = True
        return True

    @property
    def field_names(self) -> List[str]: return list(self._field_names)
    @property
    def events(self) -> List[Event]: return list(self._events)
    @property
    def reader(self) -> Reader: return self._reader

# ==========================================
# PART 8: Main Execution (CLI) - MODIFIED
# ==========================================

def main():
    parser_args = argparse.ArgumentParser(description="Recursive BBL to CSV converter")
    parser_args.add_argument("paths", nargs='+', help="Path to .bbl files or directories")
    parser_args.add_argument("-i", "--index", type=int, help="Specific log index to decode")
    
    args = parser_args.parse_args()
    
    # 1. Collect all files to process
    files_to_process = []
    
    for p in args.paths:
        if os.path.isfile(p):
            if p.lower().endswith('.bbl'):
                files_to_process.append(p)
            else:
                print(f"Skipping non-bbl file: {p}")
        elif os.path.isdir(p):
            print(f"Scanning directory: {p}")
            for root, dirs, files in os.walk(p):
                for file in files:
                    if file.lower().endswith(".bbl"):
                        full_path = os.path.join(root, file)
                        files_to_process.append(full_path)
        else:
            print(f"Error: Path not found: {p}")

    if not files_to_process:
        print("No .bbl files found.")
        return

    print(f"Found {len(files_to_process)} .bbl files to process.")
    
    # 2. ASK USER PREFERENCE
    print("-" * 40)
    user_input = input("Generate separate 'events.csv' files? (y/N) [Default: No]: ").strip().lower()
    # Logic: Only true if user explicitly types y or yes. Default (empty) is False.
    generate_events = user_input == 'y' or user_input == 'yes'
    print("-" * 40)

    # 3. Process each file
    for log_file in files_to_process:
        print(f"Processing: {log_file}")
        try:
            bbl_parser = Parser.load(log_file)
            log_count = bbl_parser.reader.log_count
            print(f"  Found {log_count} logs.")

            target_indexes = [args.index] if args.index else range(1, log_count + 1)
            
            for idx in target_indexes:
                if idx < 1 or idx > log_count: continue
                
                print(f"  > Decoding Log #{idx}...")
                bbl_parser.set_log_index(idx)
                
                # Setup output paths (same directory as input file)
                base_name, _ = os.path.splitext(log_file)
                out_csv = f"{base_name}.{idx:02d}.csv"
                out_events = f"{base_name}.{idx:02d}.events.csv"
                
                # Write Main CSV
                with open(out_csv, 'w', newline='') as f:
                    writer = csv.writer(f)
                    field_names = bbl_parser.field_names
                    if "time" not in field_names and "time (us)" not in field_names:
                        field_names = ["time"] + field_names
                    writer.writerow(field_names)
                    
                    for frame in bbl_parser.frames():
                        # Simple formatting
                        row = [f"{x:.2f}" if isinstance(x, float) else str(x) for x in frame.data]
                        writer.writerow(row)
                
                # Write Events - CONDITIONALLY
                if generate_events and bbl_parser.events:
                    with open(out_events, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(["time", "event", "data"])
                        for evt in bbl_parser.events:
                            t = evt.data.get("time", 0) if isinstance(evt.data, dict) else 0
                            writer.writerow([t, evt.type.name, evt.data])
                    print(f"    - Completed: {out_csv} AND {out_events}")
                else:
                    print(f"    - Completed: {out_csv}")
                
        except Exception as e:
            print(f"  Failed to process {log_file}: {e}")
        print("-" * 40)

if __name__ == "__main__":
    main()