# pattern_generation.py

import random
import numpy as np
import pandas as pd
import spacy
from datetime import datetime, timezone
import os
import json
from tqdm import tqdm
import sys
import requests
from sklearn.metrics.pairwise import cosine_similarity

# Configure tqdm to write to stderr for better visibility
progress_bar = lambda x, **kwargs: tqdm(x, file=sys.stderr, ncols=100, leave=True, **kwargs)

print("Loading models...")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_lg")
    print("Models loaded successfully")
except OSError:
    print("Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")
    print("Models loaded successfully")

# Initialize files
dataset_file = 'thinking_patterns.csv'
progress_file = 'generation_progress.json'
benchmark_file = 'benchmark_results.json'

# Ollama API configuration
OLLAMA_API_HOST = "http://127.0.0.1:11434"  # Adjusted to match local setup
OLLAMA_MODEL = "llama3.2:latest"  # Updated model name

# Headers for Ollama API requests
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0)',
    'Content-Type': 'application/json'
}

# Enhanced metrics tracking
metrics = {
    'agi_alignment': [],
    'architectural_coherence': [],
    'web_recognition': [],
    'vector_quality': []
}

def load_progress():
    """Load progress from file or initialize new progress"""
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
            return progress['completed_records'], progress['total_records']
    return 0, 100

def generate_random_topic():
    """Generate a random topic from predefined categories"""
    topics = {
        'technology': [
            "Artificial General Intelligence development",
            "Neural architecture search",
            "Quantum computing advances",
            "Web 3.0 and decentralization",
            "Machine consciousness theories",
            "Emergent AI behaviors"
        ],
        'science': [
            "Cognitive architecture models",
            "Brain-computer interfaces",
            "Quantum cognition theories",
            "Neuromorphic computing",
            "Artificial life evolution",
            "Computational neuroscience"
        ],
        'society': [
            "Human-AI collaboration",
            "Digital consciousness ethics",
            "Technological singularity",
            "AI governance frameworks",
            "Cognitive enhancement ethics",
            "AGI safety protocols"
        ],
        'web_patterns': [
            "Distributed intelligence systems",
            "Semantic web architectures",
            "Knowledge graph evolution",
            "Web-scale learning patterns",
            "Collective intelligence emergence",
            "Digital ecosystem dynamics"
        ]
    }
    category = random.choice(list(topics.keys()))
    topic = random.choice(topics[category])
    return topic

# Initialize thought templates with weights
thought_templates = [
    {
        'template': "Analyzing distributed systems architecture of {topic} across multiple nodes",
        'weight': 1.0
    },
    {
        'template': "Evaluating scalability patterns in {topic} with focus on web infrastructure",
        'weight': 1.0
    },
    {
        'template': "Exploring fundamental concepts of {topic}",
        'weight': 1.0
    },
    {
        'template': "Analyzing relationships between {topic} components",
        'weight': 1.0
    },
    {
        'template': "Identifying key patterns in {topic}",
        'weight': 1.0
    },
    {
        'template': "Synthesizing insights about {topic}",
        'weight': 1.0
    },
    {
        'template': "Analyzing the AGI implications of {topic}",
        'weight': 1.0
    },
    {
        'template': "Mapping the architectural components of {topic} in cognitive systems",
        'weight': 1.0
    },
    {
        'template': "Evaluating {topic} through multiple intelligence frameworks",
        'weight': 1.0
    },
    {
        'template': "Identifying emergent properties in {topic} systems",
        'weight': 1.0
    },
    {
        'template': "Modeling consciousness aspects of {topic}",
        'weight': 1.0
    },
    {
        'template': "Tracking {topic} patterns across web-scale data",
        'weight': 1.0
    },
    {
        'template': "Analyzing distributed aspects of {topic} in network structures",
        'weight': 1.0
    },
    {
        'template': "Mapping knowledge flows related to {topic} in digital ecosystems",
        'weight': 1.0
    },
    {
        'template': "Decomposing {topic} into architectural components",
        'weight': 1.0
    },
    {
        'template': "Identifying system boundaries and interfaces in {topic}",
        'weight': 1.0
    },
    {
        'template': "Evaluating scalability patterns in {topic}",
        'weight': 1.0
    },
    {
        'template': "Measuring performance metrics for {topic}",
        'weight': 1.0
    },
    {
        'template': "Establishing baseline comparisons for {topic}",
        'weight': 1.0
    },
    {
        'template': "Defining success criteria for {topic} implementation",
        'weight': 1.0
    }
]

def simulate_thinking(topic):
    """Generate a thinking pattern based on the topic"""
    total_weight = sum(t['weight'] for t in thought_templates)
    probabilities = [t['weight'] / total_weight for t in thought_templates]
    selected_template = np.random.choice(thought_templates, p=probabilities)
    pattern = selected_template['template'].format(topic=topic)
    return pattern

def check_ollama_health():
    """Check if Ollama service is running"""
    try:
        response = requests.get(OLLAMA_API_HOST, headers=headers, timeout=5)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"\nFailed to connect to Ollama service: {str(e)}")
        print("Please ensure Ollama is installed and running on port 11434")
        print("Try running 'OLLAMA_HOST=0.0.0.0:11434 ollama serve' in a terminal")
        return False

def get_llm_analysis(thought):
    """Generate analysis using LLaMA model via Ollama"""
    # Prepare the prompt
    prompt = f"Analyze the following thought pattern:\n\n\"{thought}\"\n\nConsider the following aspects:\n1. Type of thinking demonstrated\n2. Key concepts involved\n3. Potential implications\n"

    # Use Ollama to generate the analysis
    if not check_ollama_health():
        return ""
    try:
        url = f"{OLLAMA_API_HOST}/api/generate"
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()

        # Process the response line by line
        full_response = ""
        for line in response.text.splitlines():
            if line.strip():
                try:
                    data = json.loads(line)
                    if 'response' in data:
                        full_response += data['response']
                except json.JSONDecodeError:
                    continue
        return full_response.strip()
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {str(e)}")
        return ""

def get_sentence_embedding(sentence):
    """Get embedding from LLaMA via Ollama"""
    if not check_ollama_health():
        return np.zeros(4096)  # Assuming embedding size is 4096
    try:
        url = f"{OLLAMA_API_HOST}/api/embeddings"
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": sentence,
            "embedding_only": True
        }
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()

        # Process the response line by line
        embedding = None
        for line in response.text.splitlines():
            if line.strip():
                try:
                    data = json.loads(line)
                    if 'embedding' in data:
                        embedding = data['embedding']
                        break  # We have the embedding, no need to continue
                except json.JSONDecodeError:
                    continue
        if embedding is not None:
            vector = np.array(embedding, dtype=float)
            return vector
        else:
            print("Error: No embedding found in response")
            return np.zeros(4096)
    except requests.exceptions.RequestException as e:
        print(f"Error getting embeddings: {str(e)}")
        return np.zeros(4096)

def benchmark_pattern(pattern, vector):
    """Benchmark pattern quality across multiple dimensions"""
    scores = {
        'agi_alignment': evaluate_agi_alignment(pattern),
        'architectural_coherence': measure_arch_coherence(pattern),
        'web_recognition': assess_web_recognition(pattern),
        'vector_quality': evaluate_vector_quality(vector)
    }
    return scores

def evaluate_agi_alignment(pattern):
    """Evaluate pattern alignment with AGI principles"""
    agi_keywords = ['intelligence', 'learning', 'cognition', 'reasoning', 'adaptation',
                    'emergence', 'consciousness', 'scalability', 'architecture']
    doc = nlp(pattern)
    keyword_matches = sum(1 for token in doc if token.lemma_.lower() in agi_keywords)
    keyword_score = min(keyword_matches / len(agi_keywords), 1.0)
    agi_concept = "artificial general intelligence cognitive systems"
    pattern_embedding = get_sentence_embedding(pattern)
    agi_embedding = get_sentence_embedding(agi_concept)
    if np.linalg.norm(pattern_embedding) == 0 or np.linalg.norm(agi_embedding) == 0:
        similarity = 0.0
    else:
        similarity = cosine_similarity(
            [pattern_embedding],
            [agi_embedding]
        )[0][0]
    keyword_weight = 0.4
    similarity_weight = 0.6
    final_score = (keyword_score * keyword_weight) + (similarity * similarity_weight)
    return final_score

def measure_arch_coherence(pattern):
    """Measure architectural coherence of pattern"""
    doc = nlp(pattern)
    arch_principles = {
        'modularity': ['component', 'module', 'interface', 'boundary', 'part', 'element'],
        'scalability': ['scale', 'growth', 'expand', 'adapt', 'evolve', 'flexible'],
        'integration': ['connect', 'integrate', 'interact', 'communicate', 'link', 'combine'],
        'structure': ['architecture', 'framework', 'structure', 'design', 'pattern', 'system']
    }
    principle_scores = {}
    for principle, keywords in arch_principles.items():
        matches = sum(1 for token in doc if token.lemma_.lower() in keywords)
        principle_scores[principle] = min(matches / len(keywords), 1.0)
    tree_depth = max(len(list(token.ancestors)) for token in doc) if len(doc) > 0 else 0
    depth_score = min((tree_depth + 1) / 4, 1.0)
    weights = {
        'modularity': 0.3,
        'scalability': 0.2,
        'integration': 0.2,
        'structure': 0.2,
        'tree_depth': 0.1
    }
    final_score = sum(principle_scores.get(p, 0) * weights[p] for p in arch_principles.keys())
    final_score += depth_score * weights['tree_depth']
    return final_score

def assess_web_recognition(pattern):
    """Assess pattern recognition at web scale"""
    doc = nlp(pattern)
    web_characteristics = {
        'distribution': ['distributed', 'network', 'web-scale', 'global', 'system', 'cloud'],
        'connectivity': ['connected', 'linked', 'interconnected', 'networked', 'integrated', 'coupled'],
        'scalability': ['scalable', 'extensible', 'expandable', 'elastic', 'flexible', 'adaptive'],
        'data_flow': ['flow', 'stream', 'transfer', 'exchange', 'process', 'communicate']
    }
    char_scores = {}
    for char, keywords in web_characteristics.items():
        matches = sum(1 for token in doc if token.lemma_.lower() in keywords)
        char_scores[char] = min(matches / len(keywords), 1.0)
    web_entities = ['NETWORK', 'SYSTEM', 'PROTOCOL', 'DATA']
    entity_score = sum(1 for ent in doc.ents if ent.label_ in web_entities) / len(web_entities)
    weights = {
        'distribution': 0.3,
        'connectivity': 0.3,
        'scalability': 0.2,
        'data_flow': 0.15,
        'entities': 0.05
    }
    final_score = sum(char_scores.get(c, 0) * weights[c] for c in web_characteristics.keys())
    final_score += entity_score * weights['entities']
    return final_score

def evaluate_vector_quality(vector):
    """Evaluate quality of vector representation"""
    if not isinstance(vector, np.ndarray):
        try:
            vector = np.array(vector, dtype=float)
        except:
            return 0.0
    if len(vector) == 0:
        return 0.0
    magnitude = np.linalg.norm(vector)
    sparsity = np.count_nonzero(vector) / len(vector)
    variance = np.var(vector)
    ideal_magnitude = (0.5, 2.0)
    ideal_sparsity = (0.1, 0.5)
    ideal_variance = (0.01, 0.1)
    magnitude_score = 1.0 - min(abs(magnitude - np.mean(ideal_magnitude)) / np.mean(ideal_magnitude), 1.0)
    sparsity_score = 1.0 if ideal_sparsity[0] <= sparsity <= ideal_sparsity[1] else 0.5
    variance_score = 1.0 if ideal_variance[0] <= variance <= ideal_variance[1] else 0.5
    weights = {
        'magnitude': 0.4,
        'sparsity': 0.3,
        'variance': 0.3
    }
    final_score = (magnitude_score * weights['magnitude'] +
                   sparsity_score * weights['sparsity'] +
                   variance_score * weights['variance'])
    return final_score

def save_progress(completed, total):
    """Save current progress to file"""
    progress = {
        'completed_records': completed,
        'total_records': total,
        'last_update': datetime.now(timezone.utc).isoformat()
    }
    with open(progress_file, 'w') as f:
        json.dump(progress, f)

def update_template_weights(template_index, benchmark_score):
    """Update the weights of the templates based on benchmark scores"""
    # Increase weight if benchmark score is high, decrease if low
    if benchmark_score > 0.5:
        thought_templates[template_index]['weight'] *= 1.1  # Increase weight by 10%
    else:
        thought_templates[template_index]['weight'] *= 0.9  # Decrease weight by 10%
    # Ensure weights stay within reasonable bounds
    thought_templates[template_index]['weight'] = max(0.1, min(thought_templates[template_index]['weight'], 10.0))

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def main():
    try:
        print("Loading progress...")
        completed_records, total_records = load_progress()
        print(f"Starting from record {completed_records} of {total_records}")

        if not os.path.exists(dataset_file):
            print("Creating new dataset...")
            dataset = pd.DataFrame(columns=['Timestamp', 'Topic', 'Pattern', 'Vector', 'LLM_Analysis'])
        else:
            print("Loading existing dataset...")
            dataset = pd.read_csv(dataset_file)

        # Add adaptive targets and better monitoring
        initial_target = 0.3  # Start with 30%
        target_increment = 0.05  # Increase by 5% when target is reached
        max_target = 0.8  # Maximum target of 80%
        current_target = initial_target

        # Track progress history
        progress_history = {
            'accuracy': [],
            'improvements': [],
            'iterations': []
        }

        max_iterations = 100
        iteration_count = 0
        min_improvement_threshold = 0.001
        last_accuracy = 0.0  # Initialize last_accuracy

        while iteration_count < max_iterations:
            print(f"\nProcessing records {completed_records} to {total_records}...")
            for _ in progress_bar(range(completed_records, total_records)):
                topic = generate_random_topic()
                thought = simulate_thinking(topic)
                analysis = get_llm_analysis(thought)

                vector = get_sentence_embedding(thought)

                new_row = {
                    'Timestamp': datetime.now(timezone.utc).isoformat(),
                    'Topic': topic,
                    'Pattern': thought,
                    'Vector': vector.tolist(),
                    'LLM_Analysis': analysis
                }
                dataset = pd.concat([dataset, pd.DataFrame([new_row])], ignore_index=True)
                completed_records += 1

                if completed_records % 10 == 0:
                    save_progress(completed_records, total_records)
                    dataset.to_csv(dataset_file, index=False)

            dataset.to_csv(dataset_file, index=False)
            print("\nBenchmarking patterns...")
            benchmark_results = []

            for idx, row in progress_bar(dataset.iterrows(),
                                         desc="Benchmarking patterns",
                                         total=len(dataset)):
                pattern = row['Pattern']
                vector = np.array(row['Vector'], dtype=float)
                scores = benchmark_pattern(pattern, vector)
                benchmark_results.append(scores)

                # Update template weights based on the benchmark score
                for i, t in enumerate(thought_templates):
                    if t['template'].format(topic=row['Topic']) == pattern:
                        benchmark_score = np.mean([
                            scores['agi_alignment'],
                            scores['architectural_coherence'],
                            scores['web_recognition']
                        ])
                        update_template_weights(i, benchmark_score)
                        break  # Found the matching template

            metrics['agi_alignment'] = [b['agi_alignment'] for b in benchmark_results]
            metrics['architectural_coherence'] = [b['architectural_coherence'] for b in benchmark_results]
            metrics['web_recognition'] = [b['web_recognition'] for b in benchmark_results]
            metrics['vector_quality'] = [b['vector_quality'] for b in benchmark_results]

            current_accuracy = np.mean([
                np.mean(metrics['agi_alignment']),
                np.mean(metrics['architectural_coherence']),
                np.mean(metrics['web_recognition'])
            ])

            # Calculate improvement before recording progress
            improvement = current_accuracy - last_accuracy

            # Record progress
            progress_history['accuracy'].append(current_accuracy)
            progress_history['improvements'].append(improvement)
            progress_history['iterations'].append(iteration_count)

            # Adaptive target adjustment
            if current_accuracy >= current_target:
                current_target = min(current_target + target_increment, max_target)
                print(f"Target achieved! New target: {current_target:.3f}")

            # Stop if we're not improving meaningfully
            if improvement < min_improvement_threshold and iteration_count > 10:
                print(f"Stopping: No significant improvement after {iteration_count} iterations")
                print(f"Initial accuracy: {last_accuracy:.3f}")
                print(f"Final accuracy: {current_accuracy:.3f}")
                break

            # Stop if we hit max target accuracy
            if current_accuracy >= max_target:
                print(f"Success! Reached maximum target accuracy of {max_target:.3f}")
                break

            last_accuracy = current_accuracy
            iteration_count += 1
            total_records += 100  # Increase total records to generate more data

            # Add diagnostic information
            print(f"Iteration {iteration_count}: Improvement = {improvement:.4f}")
            print(f"Current Accuracy: {current_accuracy:.4f}")
            print(f"Template Weights: {[t['weight'] for t in thought_templates]}")

        print("\nSaving final benchmark results...")
        with open(benchmark_file, 'w') as f:
            json.dump(benchmark_results, f, cls=NumpyEncoder)  # Use NumpyEncoder
        print("Process completed successfully")

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        save_progress(completed_records, total_records)
        raise

if __name__ == "__main__":
    main()
