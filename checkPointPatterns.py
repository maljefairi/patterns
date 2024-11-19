import random
import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime, UTC  # Added UTC import
import os
import requests
import json
from tqdm import tqdm  # For progress bars
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

# Initialize or load the dataset and progress tracking
dataset_file = 'thinking_patterns.csv'
progress_file = 'generation_progress.json'
metrics_file = 'model_metrics.json'

# Initialize metrics tracking
metrics = {
    'accuracy': [],
    'precision': [],
    'recall': [], 
    'f1': [],
    'pattern_detection_rate': [],
    'vectorization_success_rate': []
}

# Load progress if it exists
if os.path.exists(progress_file):
    with open(progress_file, 'r') as f:
        progress = json.load(f)
    total_records = progress.get('total_records', 0)
    completed_records = progress.get('completed_records', 0)
    print(f"Resuming from previous run: {completed_records}/{total_records} records completed")
else:
    total_records = 100  # Default target
    completed_records = 0
    progress = {'total_records': total_records, 'completed_records': completed_records}

# Initialize or load the dataset
if os.path.exists(dataset_file):
    try:
        print(f"Loading existing dataset from {dataset_file}...")
        dataset = pd.read_csv(dataset_file)
        # Validate expected columns exist
        required_columns = ['Timestamp', 'Topic', 'Thought', 'Pattern', 'Vector', 'Detection_Method', 'LLM_Analysis']
        if not all(col in dataset.columns for col in required_columns):
            raise ValueError("CSV file missing required columns")
        print(f"Successfully loaded {len(dataset)} existing records")
        
        # Calculate initial metrics if dataset exists
        if len(dataset) > 0:
            # Split data for validation
            train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
            
            # Calculate pattern detection rate
            pattern_detection_rate = len(dataset[dataset['Pattern'].notna()]) / len(dataset)
            metrics['pattern_detection_rate'].append(pattern_detection_rate)
            
            # Calculate vectorization success rate
            vectorization_success = len(dataset[dataset['Vector'].notna()]) / len(dataset)
            metrics['vectorization_success_rate'].append(vectorization_success)
            
            print(f"Initial metrics calculated - Pattern detection rate: {pattern_detection_rate:.2f}, Vectorization success: {vectorization_success:.2f}")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating new empty dataset...")
        dataset = pd.DataFrame(columns=['Timestamp', 'Topic', 'Thought', 'Pattern', 'Vector', 'Detection_Method', 'LLM_Analysis'])
else:
    print("No existing dataset found. Creating new empty dataset...")
    dataset = pd.DataFrame(columns=['Timestamp', 'Topic', 'Thought', 'Pattern', 'Vector', 'Detection_Method', 'LLM_Analysis'])

def generate_random_topic():
    # Organized topics by category for better coverage
    topics = {
        'technology': [
            "Artificial intelligence and ethics",
            "Advancements in quantum computing", 
            "The influence of technology on privacy",
            "Future of robotics",
            "Blockchain technology",
            "Internet of Things"
        ],
        'science': [
            "The role of genetics in evolution",
            "Colonization of Mars",
            "Time travel theories", 
            "Dark matter mysteries",
            "Ocean exploration",
            "Quantum entanglement"
        ],
        'society': [
            "The impact of climate change on global economies",
            "Cultural impacts of globalization",
            "Exploring the depths of the human psyche",
            "The nature of consciousness",
            "Future of education",
            "Social media influence",
            "Sustainable development"
        ]
    }
    # Select random category then random topic
    category = random.choice(list(topics.keys()))
    topic = random.choice(topics[category])
    print(f"Generated topic: {topic} (Category: {category})")
    return topic

def simulate_thinking(topic):
    # Added more diverse thinking patterns
    thought_templates = [
        # Analytical patterns
        f"Conducting a systematic analysis of {topic} and its implications.",
        f"Evaluating the key factors that influence {topic}.",
        f"Examining the cause-and-effect relationships in {topic}.",
        f"Breaking down the core components of {topic}.",
        f"Measuring the quantifiable aspects of {topic}.",
        
        # Critical thinking patterns
        f"Questioning common assumptions about {topic}.",
        f"Identifying potential biases in our understanding of {topic}.",
        f"Comparing different perspectives on {topic}.",
        f"Challenging traditional views about {topic}.",
        f"Evaluating the evidence supporting {topic}.",
        
        # Creative patterns
        f"Imagining innovative solutions related to {topic}.",
        f"Connecting seemingly unrelated aspects of {topic}.",
        f"Envisioning future scenarios involving {topic}.",
        f"Brainstorming novel approaches to {topic}.",
        f"Reimagining the possibilities within {topic}.",
        
        # Reflective patterns
        f"Contemplating the deeper meaning of {topic}.",
        f"Considering how {topic} relates to human experience.",
        f"Analyzing historical patterns in the evolution of {topic}.",
        f"Exploring the philosophical implications of {topic}.",
        f"Meditating on the significance of {topic} in modern life."
    ]
    thought = random.choice(thought_templates)
    print(f"Generated thought: {thought}")
    return thought

def get_llm_analysis(thought):
    print("Requesting LLM analysis...")
    url = "http://localhost:11434/api/generate"
    # Randomize prompts for more diverse analysis
    prompts = [
        (
            "Analyze this thought pattern considering the following aspects:\n"
            "1. Type of thinking (analytical, creative, critical, etc.)\n"
            "2. Key concepts and relationships\n"
            "3. Potential implications\n"
        ),
        (
            "Evaluate this thought process and consider:\n"
            "1. Cognitive approach used\n"
            "2. Main themes and connections\n"
            "3. Future impact and consequences\n"
        ),
        (
            "Please analyze this thinking pattern:\n"
            "1. Mental framework employed\n"
            "2. Core ideas and their interconnections\n"
            "3. Possible outcomes and effects\n"
        )
    ]
    
    prompt = random.choice(prompts) + f"Thought: {thought}"
    payload = {
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)  # Added timeout
        response.raise_for_status()  # Raise exception for bad status codes
        result = response.json()
        analysis = result['response'].strip()
        print("LLM analysis completed successfully")
        return analysis
    except requests.exceptions.RequestException as e:
        error_msg = f"LLM Analysis failed: {str(e)}"
        print(error_msg)
        return error_msg
    except json.JSONDecodeError:
        error_msg = "LLM Analysis failed: Invalid response format"
        print(error_msg)
        return error_msg

def detect_pattern(thought):
    print("Detecting linguistic patterns...")
    doc = nlp(thought)
    # Enhanced pattern detection with more linguistic features
    pattern_elements = []
    
    # Extract key linguistic features
    for token in doc:
        if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']:
            # Include dependency relation for better context
            pattern_elements.append(f"{token.lemma_}_{token.dep_}")
        
    # Include named entities if present
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    if entities:
        pattern_elements.extend([f"{text}_{label}" for text, label in entities])
    
    pattern = ' '.join(pattern_elements)
    detection_method = 'Enhanced pattern detection using lemmas, dependencies, and named entities'
    print(f"Pattern detected with {len(pattern_elements)} elements")
    return pattern, detection_method

def vectorize_pattern(pattern):
    print("Vectorizing pattern...")
    try:
        # Initialize vectorizer with adjusted parameters
        vectorizer = TfidfVectorizer(
            min_df=1,
            max_df=1.0,
            stop_words='english',
            ngram_range=(1, 2)
        )
        vector = vectorizer.fit_transform([pattern]).toarray()[0]
        print(f"Pattern successfully vectorized with dimension {len(vector)}")
        return vector
    except Exception as e:
        print(f"Error in vectorization: {e}")
        raise

def add_to_dataset(dataset, topic, thought, pattern, vector, detection_method, llm_analysis):
    print("Adding new record to dataset...")
    timestamp = datetime.now(UTC).isoformat()  # Using timezone-aware datetime
    new_row = {
        'Timestamp': timestamp,
        'Topic': topic,
        'Thought': thought,
        'Pattern': pattern,
        'Vector': vector.tolist(),
        'Detection_Method': detection_method,
        'LLM_Analysis': llm_analysis
    }
    return pd.concat([dataset, pd.DataFrame([new_row])], ignore_index=True)

def save_progress(completed_records, total_records):
    progress = {
        'completed_records': completed_records,
        'total_records': total_records,
        'last_update': datetime.now(UTC).isoformat()
    }
    with open(progress_file, 'w') as f:
        json.dump(progress, f)
    
    # Save metrics
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f)

def main():
    global dataset
    try:
        print("\nStarting pattern generation process...")
        remaining_records = total_records - completed_records
        
        with tqdm(total=remaining_records, initial=completed_records, desc="Generating patterns") as pbar:
            for i in range(completed_records, total_records):
                print(f"\nProcessing record {i+1}/{total_records}...")
                topic = generate_random_topic()
                thought = simulate_thinking(topic)
                pattern, detection_method = detect_pattern(thought)
                vector = vectorize_pattern(pattern)
                llm_analysis = get_llm_analysis(thought)
                dataset = add_to_dataset(dataset, topic, thought, pattern, vector, detection_method, llm_analysis)
                
                # Calculate and update metrics every 10 records
                if (i + 1) % 10 == 0:
                    try:
                        print(f"\nCalculating metrics and saving progress...")
                        
                        # Split data for validation
                        if len(dataset) > 10:  # Ensure enough data for splitting
                            train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
                            
                            # Calculate pattern detection rate
                            pattern_detection_rate = len(dataset[dataset['Pattern'].notna()]) / len(dataset)
                            metrics['pattern_detection_rate'].append(pattern_detection_rate)
                            
                            # Calculate vectorization success rate
                            vectorization_success = len(dataset[dataset['Vector'].notna()]) / len(dataset)
                            metrics['vectorization_success_rate'].append(vectorization_success)
                            
                            print(f"Current metrics:")
                            print(f"Pattern detection rate: {pattern_detection_rate:.2f}")
                            print(f"Vectorization success rate: {vectorization_success:.2f}")
                        
                        dataset.to_csv(dataset_file, index=False)
                        save_progress(i + 1, total_records)
                        print(f"Progress saved: {i + 1}/{total_records} records")
                    except Exception as e:
                        print(f"Error saving intermediate progress: {e}")
                
                pbar.update(1)
        
        # Final save
        try:
            print(f"\nSaving final dataset to {dataset_file}...")
            dataset.to_csv(dataset_file, index=False)
            save_progress(total_records, total_records)
            print(f"Successfully saved {len(dataset)} records to {dataset_file}")
            
            # Print final metrics summary
            print("\nFinal Metrics Summary:")
            print(f"Pattern Detection Rate: {np.mean(metrics['pattern_detection_rate']):.2f}")
            print(f"Vectorization Success Rate: {np.mean(metrics['vectorization_success_rate']):.2f}")
            
        except Exception as e:
            print(f"Error saving final dataset: {e}")
            
    except Exception as e:
        print(f"Error in main execution: {e}")
        # Save progress even if there's an error
        save_progress(completed_records, total_records)

if __name__ == "__main__":
    main()