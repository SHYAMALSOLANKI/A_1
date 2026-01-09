import json
import re
import random
import os

# Configuration
INPUT_FILE = r'c:\Users\Shyamal solanki\A_1\training_data\corpus.jsonl'
OUTPUT_FILE = r'c:\Users\Shyamal solanki\A_1\training_data\pb2s_dataset.jsonl'
MAX_SAMPLES = 100000 # Limit for now to test speed/quality

def clean_latex(text):
    """
    Aggressively strips LaTeX macros to reveal the underlying Logic/Language.
    """
    # Remove document setup
    text = re.sub(r'\\documentclass\[.*?\]\{.*?\}', '', text)
    text = re.sub(r'\\usepackage\[.*?\]\{.*?\}', '', text)
    text = re.sub(r'\\begin\{.*?\}', '', text)
    text = re.sub(r'\\end\{.*?\}', '', text)
    text = re.sub(r'\\item', '', text)
    
    # Remove geometry and chaotic tags
    text = re.sub(r'\[\d+pt,.*?\]', '', text)
    text = re.sub(r'\[hmargin=.*?\]', '', text)
    
    # Remove common latex macros like \textbf{...} -> ...
    # This is a naive removal, just stripping the commands
    text = re.sub(r'\\[a-zA-Z]+\{', '', text) 
    text = re.sub(r'\}', '', text)
    
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def distort_text(text):
    """
    Creates a 'Draft' by corrupting the clean text.
    Simulates a model that knows facts but structures them poorly.
    """
    words = text.split()
    if len(words) < 5:
        return text, "Short text, potential missing context."
    
    corruption_type = random.choice(['drop_words', 'shuffle_sentence', 'truncate'])
    
    if corruption_type == 'drop_words':
        # Drop 30% of words randomly
        new_words = [w for w in words if random.random() > 0.3]
        draft = ' '.join(new_words)
        reflection = "REFLECT: The draft feels disjointed and is missing connecting words. Several key terms appear to be dropped."
        
    elif corruption_type == 'shuffle_sentence':
        # Randomly shuffle a segment
        middle = len(words) // 2
        segment = words[middle-5:middle+5]
        random.shuffle(segment)
        words[middle-5:middle+5] = segment
        draft = ' '.join(words)
        reflection = "REFLECT: The sequence of ideas is confused. Grammatical structure is broken in the middle of the response."
        
    else: # Truncate
        cutoff = int(len(words) * 0.7)
        draft = ' '.join(words[:cutoff])
        reflection = "REFLECT: The explanation cuts off abruptly. It fails to conclude the thought."
        
    return draft, reflection

def generate_pb2s_entry(raw_entry):
    """
    Takes a raw entry and converts it into the PB2S Reasoning Cycle.
    """
    # 1. Extract and Clean Truth
    raw_text = raw_entry.get('text', '')
    # The current dataset conflates User input and content. 
    # We will treat the whole clean block as the "Knowledge" the model needs to output.
    
    # Clean the noise
    clean_truth = clean_latex(raw_text)
    if len(clean_truth) < 50:
        return None # Skip garbage

    # 2. Generate Synthetic Draft (The User Prompt is technically asking about this topic)
    # We simulate a user prompt based on the first few words
    words = clean_truth.split()
    topic = ' '.join(words[:5])
    user_prompt = f"User: Explain {topic}..."
    
    # 3. Create the Corruption
    draft_text, reflection_text = distort_text(clean_truth)
    
    # 4. Formulate the PB2S Structure
    # DRAFT -> REFLECT -> REVISE -> LEARN
    
    full_sequence = (
        f"{user_prompt}\n\n"
        f"DRAFT: {draft_text}\n"
        f"{reflection_text}\n"
        f"REVISE: {clean_truth}\n"
        f"LEARNED: Contextual completeness and grammatical order are required for accurate explanation."
    )
    
    return {
        "text": full_sequence,
        "source": "synthetic_pb2s_v1",
        "type": "reasoning_trajectory"
    }

def main():
    print(f"Processing {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print("Input file not found!")
        return

    count = 0
    with open(INPUT_FILE, 'r', encoding='utf-8') as fin, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            if count >= MAX_SAMPLES:
                break
                
            try:
                entry = json.loads(line)
                new_entry = generate_pb2s_entry(entry)
                
                if new_entry:
                    fout.write(json.dumps(new_entry) + '\n')
                    count += 1
                    
                if count % 1000 == 0:
                    print(f"Generated {count} samples...")
                    
            except json.JSONDecodeError:
                continue
                
    print(f"Done. Generated {count} samples at {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
