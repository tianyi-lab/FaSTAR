import os
import json
import argparse
from PIL import Image
from fast_slow_planning import execute_fast_slow_planning
from inductive_reasoning import refine_rules_if_needed

def main(image_path, prompt_text, output_chain="Chain.json", output_image="final_output.png", alpha=0, quality_threshold=0.8):

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    llm_api_key = os.getenv("OPENAI_API_KEY")
    if not llm_api_key:
        raise ValueError("API key for OpenAI is required. Set it as an environment variable: OPENAI_API_KEY. Ensure you have access to openAI o1 model.")
    
    final_image, path, cost, quality = execute_fast_slow_planning(
        image_path, 
        prompt_text, 
        output_chain, 
        output_image, 
        alpha, 
        quality_threshold
    )
    
    if final_image:
        print(f"Total cost: {cost:.2f}")
        print(f"Final quality: {quality:.2f}")
        return final_image, path, cost, quality
    else:
        print("No final image generated.")
        return None, path, cost, quality

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute FaSTA* algorithm for image editing.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for the task.")
    parser.add_argument("--output", type=str, default="Chain.json", help="Output file for the subtask chain JSON.")
    parser.add_argument("--output_image", type=str, default="final_output.png", help="Path to save the final output image.")
    parser.add_argument("--alpha", type=float, default=0, help="Alpha parameter for cost-quality trade-off.")
    parser.add_argument("--quality_threshold", type=float, default=0.8, help="Quality threshold for VLM checks.")
    
    args = parser.parse_args()
    
    main(args.image, args.prompt, args.output, args.output_image, args.alpha, args.quality_threshold)
