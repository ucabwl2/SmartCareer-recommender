from transformers import AutoModel, AutoTokenizer
import torch
import onnx
import onnxruntime
import os
import numpy as np


class SimcseInferModel():
    def __init__(self,
                 model_name="kevinng77/TinyBERT_4L_312D_SIMCSE_finetune",

                 onnx_model_path="model.onnx",
                 ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if os.path.exists(onnx_model_path):
            self.model = self.load_onnx_model(onnx_model_path)
        else:
            torch_model = AutoModel.from_pretrained(model_name)
            self.save_onnx_model(torch_model=torch_model,
                                 model_path=onnx_model_path)
            self.model = self.load_onnx_model(onnx_model_path)
        

    def load_onnx_model(self, model_path):
        print(f">>> loading onnx model from {model_path}")
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        return onnxruntime.InferenceSession(model_path)
        
    def save_onnx_model(self, torch_model, model_path):
        torch_model.eval()
        sample_input_ids = torch.ones((1,10),dtype=torch.long)
            
        torch.onnx.export(model=torch_model,
                          args=(sample_input_ids),
                          f=model_path,
                          input_names=["input_ids"],
                          output_names=['output'],
                          dynamic_axes={'input_ids':[0,1],
                                        "output":[0]},
                          opset_version=11)
        
    def __call__(self, texts):
        """
        Args:
            texts (List[str]): Expereiences for a candicate.

        Returns:
            List[int]: List of Embeddings. Each row represent a experiences
        """
        inputs = self.tokenizer(texts, 
                                return_token_type_ids=False,
                                padding=True, 
                                truncation=True, 
                                return_tensors="np")


        output = self.model.run(None, {
            "input_ids":inputs["input_ids"].astype("int64"),
            # "attention_mask":inputs["attention_mask"].astype("int64")
        })

        return np.array(output[0])[:,0,:]
    
    
if __name__ == "__main__":
    import time

    # this is a summary
    text = [
        "Role is for existing client initiative to implement Guidewire BillingCenter (expected to complete Q1 2017) and Guidewire ClaimCenter (expected to complete Q3 2017).  Project is being overseen by client CIO.  Key people working with would include both business and IT team members from PwC, Guidewire, and client team.  Approximately 100 overall stakeholders ramping up on the program team.  Role provides leadership opportunity to work directly with client executives and shape their overall business transformation.  Day to day responsibilities include: - Leading configuration developers to implement user stories per functional requirements - Ensuring that development standards are being adhered to within workstream - Conducting code review sessions to ensure consistency among the track and also cross track - Leading detailed sprint planning for respective workstream - Participating in daily scrum meetings - Coding and configuring technical components to match requirements defined by business analysts  The following experience and education is required: - Bachelor's degree - Demonstrated experience as Guidewire BillingCenter configuration developer - Demonstrated experience in Guidewire configuration lead role (Policy, Billing, or Claims) - Basic project management skills  The following experience is preferred: - Demonstrated experience as Guidewire BillingCenter configuration lead - Guidewire BillingCenter certified - Demonstrated experience working with Agile PM tools (e.g. Rally)"
    ]
        
    # inference steps
    model = SimcseInferModel()
    time1 = time.time()
    output = model(text)

    cost_time = (time.time() - time1)*1000
    print("output shape", np.array(output).shape)
    print(f"Average time cost each step: {cost_time:.4f} ms")  # 130 ms
    print(output.shape)  # 3,312
