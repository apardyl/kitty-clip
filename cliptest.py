import clip
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Prepare the inputs
dataset = datasets.ImageFolder('archive/images', transform=preprocess)
train_set, val_set = torch.utils.data.random_split(dataset, [50470, 12617])


def logistic_regression():
    # 48%

    def get_features(dataset):
        all_features = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
                features = model.encode_image(images.to(device))

                all_features.append(features)
                all_labels.append(labels)

        return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

    # Calculate the image features
    train_features, train_labels = get_features(train_set)
    test_features, test_labels = get_features(val_set)

    # Perform logistic regression
    classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
    classifier.fit(train_features, train_labels)

    # Evaluate using the logistic regression classifier
    predictions = classifier.predict(test_features)
    accuracy = np.mean((test_labels == predictions).astype(np.float)) * 100.
    print(f"Accuracy = {accuracy:.3f}")



def zero_shot():
    # 26%

    expected_labels = []
    actual_labels = []
    for images, labels in tqdm(DataLoader(val_set, batch_size=100)):
        for image, class_id in zip(images, labels):
            image_input = image.unsqueeze(0).to(device)
            text_inputs = torch.cat([clip.tokenize(f"a photo of a {c.lower()} cat") for c in dataset.classes]).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_inputs)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values, indices = similarity[0].topk(1)
            actual_labels.append(indices[0].cpu())
            expected_labels.append(class_id)

    expected_t = torch.tensor(expected_labels)
    actual_t = torch.tensor(actual_labels)
    acc = (expected_t == actual_t).float().mean()
    print('Zero shot accuracy: {:.2f}%'.format(acc * 100))


zero_shot()
logistic_regression()
