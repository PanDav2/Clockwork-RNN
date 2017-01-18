from __future__ import division
import wave 
from numpy import abs

##rate, signal = scipy.io.wavfile.read('data/gen/bad_taste.wav')

DEFAULT_FILENAME = 'data/gen/bad_taste.wav'


class preprocess(object):
    def __init__(self, filename=DEFAULT_FILENAME):
        assert isinstance(filename, basestring), 'wrong filename type {}'.format(
            type(filename))
        self.filename = filename
        wav_file = scipy.io.wavfile.read(filename)
        print(dir)
    def normalize(self, verbose=False):
    	#temp_  = np.abs(self.signal)
    	temp_ = self.signal
    	#max_ = max(temp_)
    	print(max_)
    	#self.signal_ = self.signal/max_


def main():
	p = preprocess(DEFAULT_FILENAME)
	print(p.rate)
	p.normalize()


if __name__ == '__main__':
    main()
