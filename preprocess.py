from __future__ import division
import scipy.io.wavfile
from numpy import abs, sign,array
from collections import Counter
import matplotlib.pyplot as plt 

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
    			return value*s
    	self.nsignal = array(map(truncated,self.signal/2**encoding))
    	if verbose :
    		print(self.nsignal)
    
    def show_signal(self):
    	if self.nsignal != None:
    		signal = self.nsignal
    	else:
    		signal = self.signal
    	plt.figure()
    	plt.plot(signal,"-x")
    	plt.show()


def main():
	p = preprocess(DEFAULT_FILENAME)
	print(p.rate)
	p.normalize()
	p.show_signal()
	#print(Counter(p.nsignal).most_common()[:10])

if __name__ == '__main__':
    main()
