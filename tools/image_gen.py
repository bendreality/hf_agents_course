import json
from random import randint
from urllib import request
from smolagents import tool
#This is the ComfyUI api prompt format.

#If you want it for a specific workflow you can "enable dev mode options"
#in the settings of the UI (gear beside the "Queue Size: ") this will enable
#a button on the UI to save workflows in api format.

#keep in mind ComfyUI is pre alpha software so this format will change a bit.

#this is the one for the default workflow

prompt_text = """
{
  "3": {
    "inputs": {
      "seed": 513842954914741,
      "steps": 8,
      "cfg": 2,
      "sampler_name": "euler",
      "scheduler": "sgm_uniform",
      "denoise": 1,
      "model": [
        "4",
        0
      ],
      "positive": [
        "6",
        0
      ],
      "negative": [
        "7",
        0
      ],
      "latent_image": [
        "5",
        0
      ]
    },
    "class_type": "KSampler",
    "_meta": {
      "title": "KSampler"
    }
  },
  "4": {
    "inputs": {
      "ckpt_name": "juggernautXL_juggXILightningByRD.safetensors"
    },
    "class_type": "CheckpointLoaderSimple",
    "_meta": {
      "title": "Load Checkpoint"
    }
  },
  "5": {
    "inputs": {
      "width": [
        "11",
        0
      ],
      "height": [
        "11",
        1
      ],
      "batch_size": 1
    },
    "class_type": "EmptyLatentImage",
    "_meta": {
      "title": "Empty Latent Image"
    }
  },
  "6": {
    "inputs": {
      "text": "a cat on a tree",
      "clip": [
        "4",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "7": {
    "inputs": {
      "text": "text, watermark, sfw",
      "clip": [
        "4",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "8": {
    "inputs": {
      "samples": [
        "3",
        0
      ],
      "vae": [
        "4",
        2
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "11": {
    "inputs": {
      "resolution": "896x1152"
    },
    "class_type": "CM_SDXLResolution",
    "_meta": {
      "title": "SDXLResolution"
    }
  },
  "12": {
    "inputs": {
      "output_path": "G:\\\\Meine Ablage\\\\comfy_out\\\\",
      "filename_prefix": "ComfyUI",
      "filename_delimiter": "_",
      "filename_number_padding": 4,
      "filename_number_start": "false",
      "extension": "png",
      "dpi": 300,
      "quality": 100,
      "optimize_image": "true",
      "lossless_webp": "false",
      "overwrite_mode": "false",
      "show_history": "false",
      "show_history_by_prefix": "true",
      "embed_workflow": "true",
      "show_previews": "true",
      "images": [
        "8",
        0
      ]
    },
    "class_type": "Image Save",
    "_meta": {
      "title": "Image Save"
    }
  }
}
"""

@tool
def queue_prompt(input_prompt: str, resolution: str, image_amount: int) -> str:
    """
    This is a tool for generating images by using a text prompt.
    The prompt must be in the style of stable diffusion. Include quality and style hints.
    mention the style of the image. Is it photo real or drawn by hand or anything else?
    write a long prompt describing visual detail.

    choose a resolution that matches the image content. Here are the available resolutions:
    1024x1024
    1152x896
    896x1152
    1216x832
    832x1216
    1344x768
    768x1344
    1536x640
    640x1536

    Args:
        input_prompt: the description of the image
        resolution: one of the available resolutions as string
        image_amount: the amount of images to generate
    """
    try:

        for i in range(image_amount):
            prompt = json.loads(prompt_text)

            prompt["6"]["inputs"]["text"] = input_prompt
            prompt["3"]["inputs"]["seed"] = randint(1,654068510)
            prompt["3"]["inputs"]["resolution"] = resolution
            prompt["11"]["inputs"]["sampler_name"] = "dpm_2_ancestral"

            p = {"prompt": prompt}
            data = json.dumps(p).encode('utf-8')
            req = request.Request("http://localhost:8188/prompt", data=data)
            request.urlopen(req)

        return f"the image was generated"

    except Exception as e:
        print(f"the image was not generated. the error was:\n{e}")
        return f"the image was not generated. the error was:\n{e}"



if __name__ == "__main__":
    queue_prompt("a poster illustration of a dog riding a 60s rocket through space, perfect looking rocket, dog on top of rocket", resolution="1024x1024")

