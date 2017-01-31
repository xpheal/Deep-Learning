import sys

def file_parser(file):
	for line in file:
		items = line.strip().split("//")

		if items[0]:
			yield items[0].strip()

def load_data(scanner):
	feature_names = []
	feature_values = []
	label_values = []

	num_features = int(next(scanner))

	for i in range(num_features):
		next_feature = next(scanner).split('-')
		feature_names += [next_feature[0].strip()]
		feature_values += [[x.strip() for x in next_feature[1].split()]]

	# Assume binary labels
	label_values += [next(scanner)]
	label_values += [next(scanner)]

	num_examples = int(next(scanner))

	return num_features, num_examples, feature_names, feature_values, label_values

# return a list of examples [(label, features1, ...), ...]
def load_example(scanner, num_examples):
	examples = []
	for i in range(num_examples):
		next_example = next(scanner).split()
		examples += [[x.strip() for x in next_example[1:]]]

	return examples

# python simple_perceptron.py trainset testset
def get_args():
	if not len(sys.argv) == 3:
		print("Usage: python simple_perceptron.py <trainset> <testset>")
		sys.exit()

	return sys.argv[1], sys.argv[2]

def main():
	trainset_name, testset_name = get_args()

	with open(trainset_name, 'r') as file:
		scanner = file_parser(file)
		num_features, num_train_examples, feature_names, feature_values, label_values = load_data(scanner)
		train_set = load_example(scanner, num_train_examples)

	with open(testset_name, 'r') as file:
		scanner = file_parser(file)
		_, num_test_examples, _, _, _ = load_data(scanner)
		test_set = load_example(scanner, num_test_examples)

	print(train_set)
	print(test_set)

main()