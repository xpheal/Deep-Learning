import librosa
from os import listdir
import numpy as np
import csv

test_dir = "./LibriSpeech/test-clean"
examples = [] # To store examples
sample_rate = 44100 # Control sampling rate
n_fft = sample_rate // 4 # We are going to use 0.25s as the window
hop_length = n_fft // 4 # Hop_length is default which is n_fft // 4
num_examples_per_speaker = 150 # Number of samples to get for each speaker
use_decibel = False # Use decibel instead of magnitude if this is set to true
num_class = 6 # Number of speakers to used, must be smaller than number of possible speakers
seconds_per_example = 1
silence_threshold = 0.01 # If the amplitude in the time series does not exceed this threshold, it is not included in the examples
frequency_threshold = n_fft * 3 // 16 # A threshold calculated by checking the mean and max of STFT result

# Convert a time series into an example
def convert_series_to_example(x, label):
	D = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)
	D = D[:frequency_threshold]

	if use_decibel:
		# Use decibel
		D = librosa.amplitude_to_db(D, ref=np.max)
	else:
		# Use magnitude
		D = np.abs(D)

	if D.shape[1] != 17:
		print("HELP")

	D = np.ndarray.flatten(D)
	D = np.append(D, label)
#     D = np.transpose(D)
#     D = np.insert(D, D.shape[1], label, 1)
	
	return D

num_speakers = 0
for speaker_id in listdir(test_dir):
	# Loop through each speaker
	label = int(speaker_id)
	num_examples = 0
	curr_examples = []

	if num_speakers >= num_class:
		break
	
	for books in listdir(test_dir + "/" + speaker_id):
		# Loop through each books for speaker_id
		for audio_files in listdir(test_dir + "/" + speaker_id + "/" + books):
			# Loop through each audio for that book
			
			if audio_files.split('.')[1] != 'flac':
				# Check if its a flac file
				continue
			
			# Load time series
			file_name = test_dir + "/" + speaker_id + "/" + books + "/" + audio_files
			time_series, rate = librosa.load(file_name, sr=sample_rate)
			
			duration = len(time_series) // sample_rate
			
			sample_time_series = []
			
			hop_per_example = int(sample_rate * seconds_per_example)
			num_loops = int(duration // seconds_per_example)
			
			# First method
			start = 0
			for _ in range(num_loops - 1):
				extracted = time_series[start:start + hop_per_example]
				
				chunk = len(extracted) // 10
				start = 0
				count = 0
				for _ in range(10):
					if max(extracted[start:start + chunk]) > silence_threshold:
						count += 1
					start += chunk

				if count >= 7:
					sample_time_series += [extracted]
					
				start += hop_per_example
							
			for i in sample_time_series:
				curr_examples.append(convert_series_to_example(i, label))
				num_examples += 1
				
				if num_examples >= num_examples_per_speaker:
					break
			
			# Second method
#             curr_examples += convert_time_series_to_multiple_examples(time_series, label)
#             num_examples += D.shape[0]
	
			if num_examples >= num_examples_per_speaker:
				break
			
		if num_examples >= num_examples_per_speaker:
			# Stop for the outer loop        
			break
	
	# Second method
#     C = np.concatenate(curr_examples)
#     examples += [C[:num_examples_per_speaker]]        

	# First method
	examples += curr_examples
	num_speakers += 1
	print("Speaker {} done".format(num_speakers))

E = np.array(examples)
np.random.shuffle(E)
print(E.shape)

train_size = len(E) * 90 // 100
test_size = len(E) - train_size

train = E[:train_size]
test = E[train_size:]

print("Writing train1")
with open('train1.csv', 'w') as file:
	csv_writer = csv.writer(file)

	for i in train[:400]:
		csv_writer.writerow(i)

print("Writing train2")
with open('train2.csv', 'w') as file:
	csv_writer = csv.writer(file)

	for i in train[400:]:
		csv_writer.writerow(i)

print("Writing test")
with open('test.csv', 'w') as file:
	csv_writer = csv.writer(file)

	for i in test:
		csv_writer.writerow(i)