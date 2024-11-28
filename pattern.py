import random
import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime, timezone
import os
import json
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import sys

# Configure tqdm to write to stderr for better visibility and keep output on one line
progress_bar = lambda x, **kwargs: tqdm(x, file=sys.stderr, ncols=100, leave=True, **kwargs)

print("Loading models...")
# Load models silently
nlp = spacy.load("en_core_web_lg")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=False)
model = AutoModel.from_pretrained("bert-base-uncased", local_files_only=False)
print("Models loaded successfully")

# Initialize files
dataset_file = 'thinking_patterns.csv'
progress_file = 'generation_progress.json'
benchmark_file = 'benchmark_results.json'

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

def simulate_thinking(topic):
    """Enhanced thinking pattern generation with quality checks"""
    thought_templates = [
        f"Analyzing distributed systems architecture of {topic} across multiple nodes",
        f"Evaluating scalability patterns in {topic} with focus on web infrastructure",
        f"Exploring fundamental concepts of {topic}",
        f"Analyzing relationships between {topic} components",
        f"Identifying key patterns in {topic}",
        f"Synthesizing insights about {topic}",
        f"Analyzing the AGI implications of {topic}.",
        f"Mapping the architectural components of {topic} in cognitive systems.",
        f"Evaluating {topic} through multiple intelligence frameworks.",
        f"Identifying emergent properties in {topic} systems.",
        f"Modeling consciousness aspects of {topic}.",
        f"Tracking {topic} patterns across web-scale data.",
        f"Analyzing distributed aspects of {topic} in network structures.",
        f"Mapping knowledge flows related to {topic} in digital ecosystems.",
        f"Decomposing {topic} into architectural components.",
        f"Identifying system boundaries and interfaces in {topic}.",
        f"Evaluating scalability patterns in {topic}.",
        f"Measuring performance metrics for {topic}.",
        f"Establishing baseline comparisons for {topic}.",
        f"Defining success criteria for {topic} implementation."
    ]
    base_pattern = random.choice(thought_templates)
    
    # Add quality enhancement layers
    pattern = enhance_web_recognition(base_pattern)
    pattern = improve_architectural_coherence(pattern)
    
    return pattern

def enhance_web_recognition(pattern):
    """Improve web-scale characteristics"""
    web_keywords = ['distributed', 'scalable', 'network', 'global']
    if not any(keyword in pattern.lower() for keyword in web_keywords):
        pattern = f"In a distributed context, {pattern}"
    return pattern

def improve_architectural_coherence(pattern):
    """Enhance structural organization"""
    arch_patterns = [
        'components interact through',
        'structured approach to',
        'systematic analysis of'
    ]
    if not any(p in pattern.lower() for p in arch_patterns):
        pattern = f"Using a structured approach to {pattern}"
    return pattern

def get_llm_analysis(thought):
    """Generate LLM analysis prompts for the given thought"""
    prompts = [
        "Analyze this thought pattern considering:\n"
        "1. Type of thinking demonstrated\n"
        "2. Key concepts involved\n"
        "3. Potential implications\n",
        (
            "Analyze this thought pattern from an AGI perspective:\n"
            "1. Cognitive architecture alignment\n"
            "2. Intelligence emergence potential\n"
            "3. Scaling characteristics\n"
            "4. Safety and ethical considerations\n"
        ),
        (
            "Evaluate the architectural implications:\n"
            "1. System decomposition\n"
            "2. Interface definitions\n"
            "3. Scalability patterns\n"
            "4. Integration points\n"
        ),
        (
            "Assess web-scale recognition patterns:\n"
            "1. Distribution characteristics\n"
            "2. Network effects\n"
            "3. Emergence properties\n"
            "4. Collective intelligence aspects\n"
        )
    ]
    return random.choice(prompts)

def get_sentence_embedding(sentence):
    """Get BERT embeddings for a sentence"""
    inputs = tokenizer(sentence, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    sentence_embedding = embeddings.mean(dim=1).squeeze().numpy()
    return sentence_embedding

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
    tree_depth = max(len(list(token.ancestors)) for token in doc)
    depth_score = min((tree_depth + 1) / 4, 1.0)
    weights = {
        'modularity': 0.3,
        'scalability': 0.2,
        'integration': 0.2,
        'structure': 0.2,
        'tree_depth': 0.1
    }
    final_score = sum(principle_scores[p] * weights[p] for p in arch_principles.keys())
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
    final_score = sum(char_scores[c] * weights[c] for c in web_characteristics.keys())
    final_score += entity_score * weights['entities']
    return final_score

def evaluate_vector_quality(vector):
    """Evaluate quality of vector representation"""
    if not isinstance(vector, np.ndarray):
        try:
            vector = np.array(vector, dtype=float)
        except:
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

def analyze_low_scores(metrics):
    """Add this function to identify specific problems"""
    problem_areas = []
    if metrics['web_recognition'] < 0.1:
        problem_areas.append("Web patterns need more distributed system concepts")
    if metrics['architectural_coherence'] < 0.2:
        problem_areas.append("Patterns lack proper structure and organization")
    return problem_areas

def check_stopping_conditions(progress_history, min_improvement_threshold):
    """More sophisticated stopping conditions"""
    if len(progress_history['accuracy']) < 5:
        return False
        
    # Check recent progress
    recent_improvements = progress_history['improvements'][-5:]
    avg_improvement = np.mean(recent_improvements)
    
    # Check for stagnation
    if avg_improvement < min_improvement_threshold:
        print(f"Stopping: Average improvement ({avg_improvement:.4f}) below threshold")
        return True
        
    # Check for oscillation
    if np.std(recent_improvements) < min_improvement_threshold/2:
        print("Stopping: Progress has stabilized")
        return True
        
    return False

def analyze_performance(metrics, progress_history):
    """Generate detailed performance analysis"""
    analysis = {
        'overall_trend': np.polyfit(
            progress_history['iterations'], 
            progress_history['accuracy'], 
            1
        )[0],
        'problem_areas': [],
        'recommendations': []
    }
    
    # Analyze specific metrics
    if np.mean(metrics['web_recognition']) < 0.2:
        analysis['problem_areas'].append({
            'metric': 'web_recognition',
            'issue': 'Low web-scale characteristics',
            'recommendation': 'Increase distributed system concepts'
        })
    
    if np.mean(metrics['architectural_coherence']) < 0.3:
        analysis['problem_areas'].append({
            'metric': 'architectural_coherence',
            'issue': 'Poor structural organization',
            'recommendation': 'Enhance component relationships'
        })
    
    return analysis

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
        last_accuracy = 0

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
                scores = benchmark_pattern(row['Pattern'], row['Vector'])
                benchmark_results.append(scores)
                
            metrics['agi_alignment'] = [b['agi_alignment'] for b in benchmark_results]
            metrics['architectural_coherence'] = [b['architectural_coherence'] for b in benchmark_results]
            metrics['web_recognition'] = [b['web_recognition'] for b in benchmark_results]
            metrics['vector_quality'] = [b['vector_quality'] for b in benchmark_results]
            
            current_accuracy = np.mean([
                np.mean(metrics['agi_alignment']),
                np.mean(metrics['architectural_coherence']),
                np.mean(metrics['web_recognition'])
            ])
            
            # Record progress
            progress_history['accuracy'].append(current_accuracy)
            progress_history['improvements'].append(improvement)
            progress_history['iterations'].append(iteration_count)
            
            # Adaptive target adjustment
            if current_accuracy >= current_target:
                current_target = min(current_target + target_increment, max_target)
                print(f"Target achieved! New target: {current_target:.3f}")
            
            improvement = current_accuracy - last_accuracy
            
            # Stop if we're not improving meaningfully
            if improvement < min_improvement_threshold and iteration_count > 10:
                print(f"Stopping: No significant improvement after {iteration_count} iterations")
                print(f"Initial accuracy: {last_accuracy:.3f}")
                print(f"Final accuracy: {current_accuracy:.3f}")
                break
            
            # Stop if we hit target accuracy
            if current_accuracy >= current_target:
                print(f"Success! Reached target accuracy of {current_target:.3f}")
                break
            
            last_accuracy = current_accuracy
            iteration_count += 1
            
            # Add diagnostic information
            print(f"Iteration {iteration_count}: Improvement = {improvement:.4f}")

        print("\nSaving final benchmark results...")
        with open(benchmark_file, 'w') as f:
            json.dump(benchmark_results, f)
        print("Process completed successfully")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        save_progress(completed_records, total_records)
        raise

if __name__ == "__main__":
    main()
