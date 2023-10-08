import torch

from train import FeedForward, download_mnist_datasets

class_mapping = "0 1 2 3 4 5 6 7 8 9".split(" ")

def predict(model, inputs, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(inputs)
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]

        return predicted, expected



if __name__ == "__main__":
    # load back the model
    ffnet = FeedForward()
    state_dict = torch.load("ffnet.pth")

    # load MNIST validation dataset
    _, validation_data = download_mnist_datasets()

    # get a sample from validation dataset for inference
    input, target = validation_data[0][0], validation_data[0][1]

    # make an inference
    predicted, expected = predict(ffnet, input, target, class_mapping)

    print(f"Predicted - {predicted}\nExpectd - {expected}")
    