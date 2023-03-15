import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

INPUT_FILE		= '[Samsung] Test task 04.png'
OUTPUT_FILE		= 'Result.wav'
SAMPLE_RATE		= 44100

class VinylPlate:
	RPM					= 120
	TRACK_WIDTH			= 7			# approximate, should be: actual width <= TRACK_WIDTH <= track spacing
	CONTRAST_THRESHOLD	= 50		# do not perform needle adjustment if delta brightness is less then value
	ADJUSTMENT_ANGLE	= 0.0628	# Adjust radius 100 times per rotation

	def __init__(self, image: np.ndarray):
		# Store array
		self.image = image

		# Find center
		size_x, size_y = image.shape
		self.center_x = size_x / 2
		self.center_y = size_y / 2

		# Starting radius and angle
		self.angle = np.pi / 2		# start at top
		self.radius = (min(size_x, size_y) - self.TRACK_WIDTH) / 2 - 1

	def polar_to_xy(self, radius: float, angle: float):
		x = round(self.center_x + radius * np.cos(angle))
		y = round(self.center_y + radius * np.sin(angle))
		return x, y

	def find_radius_adjustment(self) -> float | None:
		delta_r_range = np.arange(-self.TRACK_WIDTH/2, self.TRACK_WIDTH/2, 0.2)
		color = np.zeros(delta_r_range.shape)
		for i, delta_r in enumerate(delta_r_range):
			x, y = self.polar_to_xy(self.radius + delta_r, self.angle)
			color[i] = self.image[x, y]

		# Estimate whether track is in the range
		if np.max(color) - np.min(color) < self.CONTRAST_THRESHOLD:
			return None		# cannot adjust here
		
		# print(color)

		# Find cumulative sum of energy
		energy_cumsum = np.cumsum(np.square(color))

		# Set threshold as half the total cumulative sum
		threshold = energy_cumsum[-1] / 2

		# Find index
		index = np.where(energy_cumsum >= threshold)[0][0]

		# print(f'DEBUG: find_radius_adjustment(): threshold={threshold}, index={index}')
		return delta_r_range[index]

	def place_needle(self):
		# Find track
		adj = None
		while self.radius > 0 and adj is None:
			adj = self.find_radius_adjustment()
			if adj is None:
				self.radius -= self.TRACK_WIDTH / 2
			else:
				self.radius += adj
		
		print(f'Found track at radius: {self.radius}')

	def spin(self, time: float=20.0, Fs: float=44100, debug: bool=False):
		# Create time and data vector
		time_vect = np.arange(0, time, 1/Fs)
		data_vect = np.zeros(time_vect.shape)

		# Initial state
		starting_angle = self.angle
		next_adjustment_time = starting_angle + self.ADJUSTMENT_ANGLE

		# debug
		debug_x = []
		debug_y = []

		for i, cur_time in enumerate(time_vect):
			# Read value at current position
			x, y = self.polar_to_xy(self.radius, self.angle)
			data_vect[i] = self.image[x, y]

			# Increment time and recalculate angle
			self.angle = starting_angle + cur_time * 2 * np.pi * self.RPM / 60

			# Adjust radius
			if self.angle >= next_adjustment_time:
				adj = self.find_radius_adjustment()
				if not adj is None:
					next_adjustment_time = self.angle + self.ADJUSTMENT_ANGLE
					self.radius += adj

					# DEBUG
					if debug:
						print(f'DEBUG: at time={cur_time} radius is {self.radius}')
						x, y = self.polar_to_xy(self.radius, self.angle)
						debug_x.append(x)
						debug_y.append(y)

		# DEBUG
		if debug:
			plt.figure()
			plt.plot(debug_x, debug_y)
			plt.show()

		return data_vect

# Read image
im = iio.imread(INPUT_FILE)

# Instantiate class
vinyl_inst = VinylPlate(im)

# Find initial position
vinyl_inst.place_needle()

# Spin
data = vinyl_inst.spin(20, Fs=SAMPLE_RATE) #, debug=True)

# Remove DC bias
data_dc = np.mean(data)
data -= data_dc

# Normalize
absmax = np.max(np.abs(data))
data /= absmax

# Convert to 16-bit
data_scaled = np.int16(data * 32767)

# Write output
wav.write(OUTPUT_FILE, SAMPLE_RATE, data_scaled)
