class ByteReader:
    REPR_BYTES = 4

    def __init__(self, buffer: bytes):
        self._buffer = buffer
        self._ptr = 0

    def read(self, count: int=1):
        if self._ptr + count >= len(self._buffer):
            lc, lb = self._ptr + count, len(self._buffer)
            raise IndexError(f"{lc} out of range for buffer of length {lb}")
        next_bytes, self._ptr = \
            self._buffer[self._ptr: self._ptr + count], self._ptr + count
        return next_bytes

    def get_ptr(self):
        return self._ptr

    def set_ptr(self, ptr: int):
        if ptr < 0 or ptr >= len(self._buffer):
            message = f"{ptr} out of range for buffer of length {self._buffer}"
            raise IndexError(message)
        self._ptr = ptr

    def seek_ptr(self, offset: int):
        if 0 <= self._ptr + offset < len(self._buffer):
            self._ptr += offset

    def __repr__(self):
        return self._buffer[self._ptr:].hex() \
            if self._ptr >= len(self._buffer) + self.REPR_BYTES \
            else self._buffer[self._ptr: self._ptr + self.REPR_BYTES].hex()


class BitReader:
    def __init__(self, stream: ByteReader, bs: int=1):
        self._stream: ByteReader = stream
        self._count: int = -1
        self._buffer: int = None
        self._buffer_size = bs

    def _refill_buffer(self):
        read_bits = 0
        self._buffer = self._buffer >> self._count if self._count >= 0 else 0

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

    def next_bit(self):
        if self._count == -1:
            self._refill_buffer()
        self._count, bit = self._count - 1, (self._buffer >> self._count) & 1
        return bit

    def get_bit_ptr(self):
        return self._count

    def get_pos(self):
        return self._stream.get_ptr(), self.get_bit_ptr()

    def done(self):
        if self._count < 0:
            return
        self._stream.seek_ptr(-(self._count + 1 % 8))

    def __del__(self):
        self.done()
