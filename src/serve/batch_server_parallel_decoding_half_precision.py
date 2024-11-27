import torch
from concurrent.futures import ThreadPoolExecutor
import timm
from PIL import Image
import io
import litserve as ls
import base64
import hydra
from omegaconf import DictConfig
import os
import numpy as np
import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.s3_utility import download_model_from_s3, read_s3_file
from torchvision import transforms
#download_model_from_s3('best_model.ckpt','pytorch-model-emlov4','bird_200_vit_tiny','checkpoints/bird_200_vit_tiny/')
# Define precision globally - can be changed to torch.float16 or torch.bfloat16
precision = torch.bfloat16

class ImageClassifierAPI(ls.LitAPI):
    def __init__(self, context=None):
        print(f"1. Init called")
        super().__init__()
        self.context = context

    def setup(self, device):
        print(f"2. set up called")
        """Initialize the model and necessary components"""
        self.device = device
        self.cfg = self.context  # Hydra config passed through context
        self.pool = ThreadPoolExecutor(min(os.cpu_count(), 8))
        # Create checkpoints directory if it doesn't exist
        os.makedirs(f"checkpoints/{self.cfg.batch_deployment.name}", exist_ok=True)
        os.makedirs(f"labels/{self.cfg.batch_deployment.name}", exist_ok=True)
        
        # Updated download_model_from_s3 call with output_location
        download_model_from_s3(
            local_file_name=self.cfg.ckpt_path.split(os.path.sep)[-1],
            bucket_name=self.cfg.batch_deployment.s3_model_bucket_location,
            s3_folder=self.cfg.batch_deployment.s3_model_bucket_folder_location,
            output_location=f"checkpoints/{self.cfg.batch_deployment.name}"
        )
        # Download and load labels
        self.labels = read_s3_file(
            file_name=self.cfg.batch_deployment.s3_labels_file_name,
            bucket_name=self.cfg.batch_deployment.s3_labels_bucket_location,
            s3_folder=self.cfg.batch_deployment.s3_labels_bucket_folder_location
        ).strip().split('\n')
        
        # Load checkpoint to get stored parameters
        checkpoint = torch.load(self.cfg.ckpt_path, map_location=device)
        # Create model using base_model from config and checkpoint parameters
        model_name = checkpoint['hyper_parameters']['base_model'] or self.cfg.batch_deployment.name
        num_classes = checkpoint['hyper_parameters']['num_classes'] or len(self.labels)
        self.model = timm.create_model(
            model_name=model_name,  
            num_classes=num_classes,
            pretrained=checkpoint['hyper_parameters']['pretrained']
        )
        # Remove 'model.' prefix from state dict keys
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('model.', '')
            new_state_dict[new_key] = state_dict[key]
        
        # Load the modified state dict
        self.model.load_state_dict(new_state_dict)
        # Create model and move to appropriate device with specified precision
        self.model = self.model.to(device).to(precision)
        self.model.eval()

        # Replace the timm transforms with your test transforms
        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        print("Set up Done!")
        # Remove or comment out the timm transforms
        # data_config = timm.data.resolve_model_data_config(self.model)
        # self.transforms = timm.data.create_transform(**data_config, is_training=False)

    def decode_request(self, request):
        print("3. decoding request")
        """Convert base64 encoded image to tensor"""
        image_bytes = request.get("image")
        if not image_bytes:
            raise ValueError("No image data provided")
        return image_bytes
        
    def batch(self, inputs):
        """
        Process batch and multiple inputs using parallel processing
        """
        print("4. Batch fn")
        def process_single_image(image_bytes):
            # Decode base64 string to bytes
            img_bytes = base64.b64decode(image_bytes)
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(img_bytes))
            # Transform image to tensor
            return self.transforms(image)
        batched_tensor = []
        inputs = list(self.pool.map(process_single_image, inputs))
         # stacked all tensor into a batch and to specified precision
        return torch.stack(inputs).to(self.device).to(precision)

    
    @torch.no_grad()
    def predict(self, x):
        """Run inference on the input batch"""
        print("5. Predict ",type(x))
        if type(x)==str:
            batched_tensor = []
            img_bytes = base64.b64decode(x)
        
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(img_bytes))
            # Convert to tensor and move to device
            tensor = self.transforms(image)
            batched_tensor.append(tensor)
            x = torch.stack(batched_tensor).to(self.device).to(precision)


        outputs = self.model(x)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        return probabilities

    def unbatch(self, output):
        """Split batch output into individual predictions"""
        print("6 UnBatch . ")
        return  [output[i] for  i in range(len(output))]

    def encode_response(self, output):
        """Convert model output to API response"""
        print(f"7. encode_response {self.context.get('max_batch_size')}")
        # Get top 5 predictions
        if self.context.get('max_batch_size',1)==1:
            output = output[0]
        probs, indices = torch.topk(output, k=5)
        return {
            "predictions": [
                {
                    "label": self.labels[idx.item()],
                    "probability": prob.item()
                }
                for prob, idx in zip(probs, indices)
            ]
        }

@hydra.main(version_base=None, config_path="../../configs", config_name="batch_deployment")
def main(cfg: DictConfig):
    # Initialize API with context
    api = ImageClassifierAPI(context=cfg)
    # Remove context from LitServer initialization
    server = ls.LitServer(
        api,
        accelerator = cfg.accelerator,
        max_batch_size = cfg.max_batch_size,
        batch_timeout = cfg.batch_timeout,
        workers_per_device = cfg.workers_per_device
    )
    server.run(port=8000)

if __name__ == "__main__":
    main()


