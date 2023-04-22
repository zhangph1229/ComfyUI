import json
from urllib import request, parse
import random

# this is the ComfyUI api prompt format. If you want it for a specific workflow you can copy it from the prompt section
# of the image metadata of images generated with ComfyUI
# keep in mind ComfyUI is pre alpha software so this format will change a bit.

# this is the one for the default workflow
# ['euler', 'euler_ancestral', 'heun', 'dpm_2', 'dpm_2_ancestral', 'lms', 'dpm_fast', 'dpm_adaptive', 'dpmpp_2s_ancestral', 'dpmpp_sde', 'dpmpp_2m', 'ddim', 'uni_pc', 'uni_pc_bh2']
prompt_text = """
{
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "cfg": 7,
            "denoise": 1,
            "latent_image": [
                "5",
                0
            ],
            "model": [
                "10",
                0
            ],
            "negative": [
                "7",
                0
            ],
            "positive": [
                "6",
                0
            ],
            "sampler_name": "dpm_2_ancestral",
            "scheduler": "normal",
            "seed": 2435856907,
            "steps": 50
        }
    },
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {
            "ckpt_name": "chilloutmix_NiPrunedFp32Fix.safetensors"
        }
    },
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "batch_size": 1,
            "height": 1024,
            "width": 640
        }
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": [
                "10",
                1
            ],
            "text": "((masterpiece)),(((bestquality))),illustration,1girl,maturefemale,smallbreast,beautifuldetailedeyes,longsleeves,hoodie,frills,extremelydetailedCGunity8kwallpaper,Loong,dragonbackground,loongbackground,gamecg,depthoffield,Capehood <lora:Dream:0.65>"
        }
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": [
                "10",
                1
            ],
            "text": "(((simple background))),monochrome ,lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, lowres, bad anatomy, bad hands, text, error, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, ugly,pregnant,vore,duplicate,morbid,mut ilated,tran nsexual, hermaphrodite,long neck,mutated hands,poorly drawn hands,poorly drawn face,mutation,deformed,blurry,bad anatomy,bad proportions,malformed limbs,extra limbs,cloned face,disfigured,gross proportions, (((missing arms))),(((missing legs))), (((extra arms))),(((extra legs))),pubic hair, plump,bad legs,error legs,username,blurry,bad feet"
        }
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": [
                "3",
                0
            ],
            "vae": [
                "4",
                2
            ]
        }
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": "model_1_task_id_10000_4",
            "images": [
                "8",
                0
            ]
        }
    }, 
    "10": {
      "class_type": "LoraLoader",
      "inputs": {
      "lora_name": "cuteGirlMix4_v10.safetensors",
      "strength_model":0.8,
      "strength_clip":0.8,
        "model":[
            "4",
            0 
        ], 
        "clip":[
            "4", 
            1
        ]
      }
    }
}
"""

url = "http://47.100.221.170:8188/prompt"


def queue_prompt(prompt):
    p = {"prompt": prompt}
    data = json.dumps(p).encode('utf-8')
    req = request.Request(url, data=data)
    return request.urlopen(req)


prompt = json.loads(prompt_text)
# set the text prompt for our positive CLIPTextEncode
prompt["6"]["inputs"]["text"] = "masterpiece best quality man"

# set the seed for our KSampler node
prompt["3"]["inputs"]["seed"] = 5

output = queue_prompt(prompt)
print(output.status)
