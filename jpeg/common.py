from typing import Optional


class ByteReader:
    REPR_BYTES = 4

    def __init__(self, buffer: bytes):
        self._buffer = buffer
        self._ptr = 0

    def read(self, count: int=1):
        if self._ptr + count >= self.size:
            self._raise_index_error(self._ptr + count)
        next_bytes, self._ptr = \
            self._buffer[self._ptr: self._ptr + count], self._ptr + count
        return next_bytes

    def get_ptr(self):
        return self._ptr

    def set_ptr(self, pos: int):
        if 0 <= pos < self.size:
            self._ptr = pos
        else:
            self._raise_index_error(pos)

    def seek_ptr(self, offset: int):
        if 0 <= self._ptr + offset < self.size:
            self._ptr += offset
        else:
            self._raise_index_error(self._ptr + offset)

    def peek(self, count: int):
        if 0 <= self._ptr + count < self.size:
            indices = (self._ptr, self._ptr + count)
            return self._buffer[min(indices): max(indices)]
        else:
            self._raise_index_error(self._ptr + count)

    def _raise_index_error(self, index: int):
        message = f"{index} out of range for buffer of length {self.size}"
        raise IndexError(message)

    @property
    def size(self) -> int:
        return len(self._buffer)

    def __repr__(self):
        return self._buffer[self._ptr:].hex() \
            if self._ptr >= self.size + self.REPR_BYTES \
            else self._buffer[self._ptr: self._ptr + self.REPR_BYTES].hex()


class BitReader:
    def __init__(self, stream: ByteReader, bs: int=1):
        self._stream: ByteReader = stream
        self._count: int = -1
        self._buffer: int = None
        self._buffer_size = bs
        self._refill_buffer()

    def _refill_buffer(self, buffer: Optional[bytes]=None):
        """Internal method that reads next bytes from the stream

        Parameters
        ----------
        buffer : bytes, optional
            Optionally allows to specify pre-parsed buffer so as to avoid
            repetition of sanity checks (b'\xff\x00')

        """
        read_bits = 0
        self._buffer = self._buffer >> self._count if self._count >= 0 else 0
        if buffer is None:
            for _ in range(self._buffer_size):
                try:
                    next_byte = self._stream.read(1)[0]
                    if next_byte == 0xFF:
                        temp_byte = self._stream.read(1)[0]
                        if temp_byte != 0:
                            self._stream.seek_ptr(-2)
                            break
                    self._buffer = (self._buffer << 8) + next_byte
                    read_bits += 8
                except IndexError:
                    break
            self._count += read_bits
        else:
            found_ff = False
            for byte in buffer:
                if found_ff:
                    found_ff = False
                    continue
                if byte == 0xFF:
                    found_ff = True
                self._buffer = (self._buffer << 8) + byte
                read_bits += 8
            self._count += read_bits

    def next_bit(self):
        if self._count == -1:
            self._refill_buffer()
        if self._count == -1:
            return None
        self._count, bit = self._count - 1, (self._buffer >> self._count) & 1
        return bit

    def get_bit_ptr(self):
        return self._count

    def set_bit_ptr(self, bit_pos: int):
        self._count = bit_pos
        return self

    def get_pos(self):
        return self._stream.get_ptr(), self.get_bit_ptr()

    def set_ptr(self, pos: tuple):
        stream_pos, bit_pos = pos[0], pos[1]
        self._stream.set_ptr(stream_pos)

        _buffer_start, _ff_count, _look_behind = \
            stream_pos - self._buffer_size, 0, 1
        _next_buffer = self._stream.peek(-self._buffer_size - _look_behind)
        while _next_buffer.count(b"\xFF\x00") != _ff_count:
            _ff_count, _look_behind = _ff_count + 1, _look_behind + 1
            _next_buffer = self._stream.peek(
                -self._buffer_size - _look_behind)

        self._refill_buffer(_next_buffer[1:])
        self._count = bit_pos
        return self

    def done(self):
        if self._count < 0:
            return self
        self._stream.seek_ptr(-(self._count + 1 % 8))
        return self

    def __del__(self):
        self.done()
