#### Titanic Survival Rate Calculator Based on Pre Existing Data

# from tflearn.datasets import titanic
# titanic.download_dataset('titanic_dataset.csv')

# from tflearn.data_utils import load_csv
# data, labels = load_csv('titanic_dataset.csv', target_column=0,
#                         categorical_labels=True, n_classes=2)


# def preprocess(passengers, columns_to_delete):
#     for column_to_delete in sorted(columns_to_delete, reverse=True):
#         [passenger.pop(column_to_delete) for passenger in passengers]
#     for i in range(len(passengers)):
#         passengers[i][1] = 1. if data[i][1] == 'female' else 0.
#     return np.array(passengers, dtype=np.float32)

# to_ignore=[1, 6]

# data = preprocess(data, to_ignore)


# net = tflearn.input_data(shape=[None, 6])
# net = tflearn.fully_connected(net, 32)
# net = tflearn.fully_connected(net, 32)
# net = tflearn.fully_connected(net, 2, activation='softmax')
# net = tflearn.regression(net)


# model = tflearn.DNN(net)

# model.fit(data, labels, n_epoch=50, batch_size=40, show_metric=True)


# passengerone = [1, 'passenger two', 'female', 17, 2, 2, 'N/A', 100.0000]
# passengertwo = [3, 'passenger one', 'male', 19, 0, 0, 'N/A', 5.0000]

# passengerone, passengertwo = preprocess([passengerone, passengertwo], to_ignore)

# pred = model.predict([passengerone, passengertwo])

# print("Passenger One Surviving Rate:", pred[0][1])
# print("Passenger Two Surviving Rate:", pred[1][1])