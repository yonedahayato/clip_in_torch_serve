from abc import ABC
import base64
import clip
from io import BytesIO
from PIL import Image
import torch
from ts.torch_handler.base_handler import BaseHandler

class CipClassifierHandler(BaseHandler, ABC):
    def __init__(self):
        print("\n== __init__ ==")
        super(CipClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        print("\n== initialize ==")
        self.device = "cpu"
        model, preprocess = clip.load("ViT-B/32", device=self.device)

        self.model = model
        self.preprocesser = preprocess

        self.initialized = True

    def preprocess(self, requests):

        def base64_to_pil(image):

            if isinstance(image, str):
                if "base64," in image:
                    # DARA URI の場合、data:[<mediatype>][;base64], を除く
                    image = image.split(",")[1]

                image = image.replace("-", "+")
                image = image.replace("_", "/")
                image = base64.b64decode(image)

            if isinstance(image, (bytearray, bytes)):

                try:
                    image = Image.open(BytesIO(image))

                except:
                    image = image.decode("ascii")
                    image = base64.b64decode(image)
                    image = Image.open(BytesIO(image))

            return image

        print("\n== preprocess ==")
        print(requests)

        inputs = requests[0]["body"]
        print(inputs)

        self.tags = inputs["tags"]
        image = inputs["image"]
        image = base64_to_pil(image)
        self._image = image

        input_ids_batch = [image]
        return input_ids_batch

    def inference(self, input_batch):
        print("\n== inference ==")
        print(input_batch)

        image = input_batch[0]
        image = self.preprocesser(image).unsqueeze(0).to(self.device)

        tags = [f"a {tag}" for tag in self.tags]
        print(tags)
        tags = clip.tokenize(tags).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(tags)

            logits_per_image, logits_per_text = self.model(image, tags)
            print("\n# image と text の類似度")
            print(logits_per_image)

            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            print("\n# soft max で確率化")
            print(probs)

        inferences = [{t : str(p) for t, p in zip(self.tags, probs[0])}]
        print(inferences)

        return inferences

    def postprocess(self, inference_output):
        print("\n== postprocess ==")

        return inference_output
