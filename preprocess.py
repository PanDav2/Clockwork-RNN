from __future__ import division
import scipy.io.wavfile
from numpy import abs, sign

##rate, signal = scipy.io.wavfile.read('data/gen/bad_taste.wav')

DEFAULT_FILENAME = 'data/gen/bad_taste.wav'


class preprocess(object):
    def __init__(self, filename=DEFAULT_FILENAME):
        assert isinstance(filename, basestring), 'wrong filename type {}'.format(
            type(filename))
        self.filename = filename
        self.rate,self.signal = scipy.io.wavfile.read(filename)

    def normalize(self, encoding = 16,verbose=False):
    	
    	def truncated(value):
    		s = sign(value)
    		value = abs(value)
    		if value>1:
    			return 1*s
    		else :
    			return value
    	self.nsignal = np.array(map(truncated,self.signal/2**encoding))
    	if verbose :
    		print(self.nsignal)

def main():
	p = preprocess(DEFAULT_FILENAME)
	print(p.rate)
	p.normalize()

if __name__ == '__main__':
    main()
