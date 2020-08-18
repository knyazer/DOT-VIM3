def crc8(msg):
	crc = 0xFF
	for x in msg:
		crc ^= x
		for j in range(8):
			if (crc & 0x80): crc = ((crc << 1) ^ 0x31) % 256
			else: crc <<= 1
	return crc
