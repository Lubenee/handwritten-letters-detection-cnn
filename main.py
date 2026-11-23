from latin_letter_dataset import get_train_set, get_test_set, LETTERS

x_train, y_train = get_train_set()
x_test, y_test = get_test_set()

print(len(LETTERS))