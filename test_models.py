#!/usr/bin/env python3

import os
import sys
import subprocess
import time
import pandas as pd
from datetime import datetime
import shutil
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("model_evaluation.log")
    ]
)
logger = logging.getLogger(__name__)

# Configuration - Primary recommended models
PRIMARY_MODELS = [
    "deepseek-r1:7b",
    "openthinker:7b",
    "phi4:14b",
    "qwen2.5:7b",
    "smallthinker:3b"
]

# Alternative models worth trying
ALTERNATIVE_MODELS = [
    "deepscaler:1.5b",  # Very small but powerful on reasoning
    "llama3.1:8b",      # Strong general capabilities
    "mistral-nemo:12b", # Long context window
    "phi3:3.8b"         # Lightweight with good reasoning
]

COMMUNITY = "thisiscere"
CONFIG_PATH = "open_source_examples/model_config.yaml"
BACKUP_CONFIG = "open_source_examples/model_config.yaml.bak"
DATA_PATH = f"data/groups/{COMMUNITY}/messages_{COMMUNITY}.csv"
RESULTS_FILE = f"comprehensive_model_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

def check_ollama_running():
    """Check if Ollama server is running."""
    try:
        subprocess.run(["ollama", "list"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except:
        logger.error("Error: Ollama is not running. Start it with 'ollama serve'")
        return False

def pull_model(model_name):
    """Pull the model from Ollama."""
    logger.info(f"Pulling model: {model_name}...")
    try:
        subprocess.run(["ollama", "pull", model_name], check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error pulling model {model_name}: {e}")
        return False

def run_model_test(model_name):
    """Run the model test by modifying the config file."""
    logger.info(f"Running test with model: {model_name}...")
    try:
        # First, load the current config
        with open(CONFIG_PATH, 'r') as f:
            import yaml
            config = yaml.safe_load(f)

        # Extract base model name without tag
        base_model_name = model_name.split(':')[0]

        # Update the model name in the config
        config['model']['name'] = base_model_name

        # Write the updated config back
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        # Verify model is available in Ollama
        try:
            # List available models
            result = subprocess.run(["ollama", "list"], stdout=subprocess.PIPE, text=True)
            models_output = result.stdout

            # Check if model is in the list
            if base_model_name not in models_output:
                logger.warning(f"Model {base_model_name} not found in Ollama list after pulling. Double-checking...")
                # Try listing as original name with tags
                if model_name not in models_output:
                    logger.error(f"Model not found in Ollama after pulling. Available models:\n{models_output}")
                    return False
        except Exception as e:
            logger.warning(f"Couldn't verify model availability: {e}")

        # Run the model playground with the updated config
        cmd = ["python", "open_source_examples/model_playground.py", DATA_PATH]
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        logger.error(f"Error running model {model_name}: {e}")
        return False

def run_conversation_evaluation():
    """Run conversation clustering evaluation."""
    logger.info("Running conversation clustering evaluation...")
    try:
        cmd = ["python", "conversation_metrics.py", f"data/groups/{COMMUNITY}"]
        subprocess.run(cmd, check=True)

        # Find the metrics file
        metrics_path = os.path.join(f"data/groups/{COMMUNITY}", f"metrics_conversations_{COMMUNITY}.csv")
        if os.path.exists(metrics_path):
            return pd.read_csv(metrics_path)
        else:
            logger.error(f"Metrics file not found: {metrics_path}")
            return None
    except Exception as e:
        logger.error(f"Error running conversation evaluation: {e}")
        return None

def run_spam_evaluation():
    """Run spam detection evaluation if GT file exists."""
    logger.info("Running spam detection evaluation...")
    spam_gt_path = os.path.join(f"data/groups/{COMMUNITY}", f"GT_spam_{COMMUNITY}.csv")

    if not os.path.exists(spam_gt_path):
        logger.warning(f"Spam ground truth file not found: {spam_gt_path}")
        return None

    try:
        cmd = ["python", "spam_metrics.py", f"data/groups/{COMMUNITY}"]
        subprocess.run(cmd, check=True)

        # Find the metrics file (using most recent if multiple exist)
        spam_metrics_files = [f for f in os.listdir(f"data/groups/{COMMUNITY}")
                             if f.startswith(f"metrics_spam_detection_{COMMUNITY}")]

        if not spam_metrics_files:
            logger.error("No spam metrics file found")
            return None

        latest_file = max(spam_metrics_files,
                         key=lambda x: os.path.getmtime(os.path.join(f"data/groups/{COMMUNITY}", x)))
        metrics_path = os.path.join(f"data/groups/{COMMUNITY}", latest_file)

        return pd.read_csv(metrics_path)
    except Exception as e:
        logger.error(f"Error running spam evaluation: {e}")
        return None

def run_topic_evaluation():
    """Run topic labeling evaluation if ANTHROPIC_API_KEY is available."""
    logger.info("Running topic labeling evaluation...")

    if not os.getenv('ANTHROPIC_API_KEY'):
        logger.warning("ANTHROPIC_API_KEY not set - skipping topic evaluation")
        return None

    try:
        cmd = ["python", "evaluate_topics.py", f"data/groups/{COMMUNITY}"]
        subprocess.run(cmd, check=True)

        # Find the metrics file (using most recent if multiple exist)
        topic_metrics_files = [f for f in os.listdir(f"data/groups/{COMMUNITY}")
                              if f.startswith(f"metrics_topics_{COMMUNITY}")]

        if not topic_metrics_files:
            logger.error("No topic metrics file found")
            return None

        latest_file = max(topic_metrics_files,
                         key=lambda x: os.path.getmtime(os.path.join(f"data/groups/{COMMUNITY}", x)))
        metrics_path = os.path.join(f"data/groups/{COMMUNITY}", latest_file)

        return pd.read_csv(metrics_path)
    except Exception as e:
        logger.error(f"Error running topic evaluation: {e}")
        return None

def combine_results(model_name, conversation_df, spam_df, topic_df):
    """Combine results from all evaluations for a model."""
    results = {"model": model_name}

    # Extract base model name for matching
    base_model_name = model_name.split(':')[0]

    # Extract conversation metrics
    if conversation_df is not None:
        # Safe filtering - check if column exists and is string type first
        if 'model' in conversation_df.columns and conversation_df['model'].dtype == 'object':
            model_conv_results = conversation_df[conversation_df['model'].str.contains(base_model_name, case=False, na=False)]
            if not model_conv_results.empty:
                results["ari_score"] = model_conv_results['ari'].values[0]
                results["messages_processed"] = model_conv_results['n_messages'].values[0]
        else:
            logger.warning(f"Couldn't filter conversation DataFrame for model {base_model_name}")

    # Extract spam metrics - similar safe filtering for other DataFrames
    if spam_df is not None:
        if 'model' in spam_df.columns and spam_df['model'].dtype == 'object':
            model_spam_results = spam_df[spam_df['model'].str.contains(base_model_name, case=False, na=False)]
            if not model_spam_results.empty:
                results["spam_accuracy"] = model_spam_results['accuracy'].values[0]
                results["spam_precision"] = model_spam_results['precision'].values[0]
                results["spam_recall"] = model_spam_results['recall'].values[0]
                results["spam_f1"] = model_spam_results['f1'].values[0]

    # Extract topic metrics with same safe filtering approach
    if topic_df is not None:
        if 'model' in topic_df.columns and topic_df['model'].dtype == 'object':
            model_topic_results = topic_df[
                (topic_df['model'].str.contains(base_model_name, case=False, na=False)) &
                (topic_df['topic'] == 'AVERAGE')
            ]
            if not model_topic_results.empty:
                results["topic_info_density"] = model_topic_results['information_density'].values[0]
                results["topic_redundancy"] = model_topic_results['redundancy'].values[0]
                results["topic_relevance"] = model_topic_results['relevance'].values[0]
                results["topic_efficiency"] = model_topic_results['efficiency'].values[0]
                results["topic_overall"] = model_topic_results['overall'].values[0]

    return results

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Evaluate multiple models on conversation analysis tasks")
    parser.add_argument("--models", nargs='+', help="Specific models to test")
    parser.add_argument("--primary-only", action="store_true", help="Only test primary recommended models")
    parser.add_argument("--alternative-only", action="store_true", help="Only test alternative models")
    args = parser.parse_args()

    if not check_ollama_running():
        sys.exit(1)

    # Backup the original config file
    shutil.copy2(CONFIG_PATH, BACKUP_CONFIG)
    logger.info(f"Original config backed up to {BACKUP_CONFIG}")

    # Determine which models to test
    models_to_test = []
    if args.models:
        models_to_test = args.models
    elif args.primary_only:
        models_to_test = PRIMARY_MODELS
    elif args.alternative_only:
        models_to_test = ALTERNATIVE_MODELS
    else:
        models_to_test = PRIMARY_MODELS + ALTERNATIVE_MODELS

    all_results = []

    # Test the models
    for model in models_to_test:
        logger.info(f"\n{'='*60}\nTesting model: {model}\n{'='*60}")

        if pull_model(model) and run_model_test(model):
            # Run all three evaluations
            conversation_results = run_conversation_evaluation()
            time.sleep(1)  # Brief pause between evaluations

            spam_results = run_spam_evaluation()
            time.sleep(1)

            topic_results = run_topic_evaluation()

            # Combine results
            combined_results = combine_results(model, conversation_results, spam_results, topic_results)
            if combined_results:
                # Add model category
                if model in PRIMARY_MODELS:
                    combined_results['category'] = 'Primary Recommendation'
                else:
                    combined_results['category'] = 'Alternative Option'

                all_results.append(combined_results)
                logger.info(f"\nResults for {model}:")
                logger.info(combined_results)

        time.sleep(5)  # Pause between models

    # Restore the original config
    shutil.copy2(BACKUP_CONFIG, CONFIG_PATH)
    logger.info(f"Original config restored from {BACKUP_CONFIG}")

    # Save combined results
    if all_results:
        final_df = pd.DataFrame(all_results)
        final_df.to_csv(RESULTS_FILE, index=False)
        logger.info(f"\nFinal comprehensive results saved to {RESULTS_FILE}")

        # Print summary focusing on ARI score (conversation clustering)
        logger.info("\n===== MODEL CONVERSATION CLUSTERING SUMMARY =====")
        if 'ari_score' in final_df.columns:
            sorted_results = final_df.sort_values(by='ari_score', ascending=False)
            summary_columns = ['model', 'ari_score', 'messages_processed', 'category']
            columns_to_show = [col for col in summary_columns if col in sorted_results.columns]
            logger.info(sorted_results[columns_to_show])

            # Find the best model
            if not sorted_results.empty:
                best_model = sorted_results.iloc[0]
                logger.info(f"\n✅ BEST MODEL: {best_model['model']} with ARI score of {best_model['ari_score']:.3f}")
                logger.info(f"   Improvement over baseline: {(best_model['ari_score'] - 0.219):.3f} points")
        else:
            logger.info("No ARI scores available in results")
    else:
        logger.error("No results were collected")

if __name__ == "__main__":
    main()
