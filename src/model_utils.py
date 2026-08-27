import torch


CLASS_NAMES = ["cats", "dogs"]


def predict_class(model, image_tensor):
    """
    Perform inference using the supplied model.

    Parameters
    ----------
    model : torch.nn.Module
        Trained CNN model.

    image_tensor : torch.Tensor
        Image tensor with shape (1, 3, 224, 224).

    Returns
    -------
    label : str
        Predicted class.

    probabilities : dict
        Probability for each class.
    """

    model.eval()

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

    predicted_index = torch.argmax(
        probabilities
    ).item()

    label = CLASS_NAMES[predicted_index]

    probability_dict = {
        CLASS_NAMES[i]: float(probabilities[i])
        for i in range(len(CLASS_NAMES))
    }

    return label, probability_dict