#!/usr/bin/env python3

import os
import sys
import subprocess
import time
import pandas as pd
import re
import traceback
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

        # Store original model name (for result matching)
        original_model_name = model_name

        # Adjust the model name format for Ollama
        # Some models need the tag removed to work properly
        if ":" in model_name:
            base_model = model_name.split(':')[0]

            # Use simplified name for models that need it
            if base_model in ["phi3", "deepseek-r1", "smallthinker", "openthinker"]:
                model_name = base_model
                logger.info(f"Using simplified model name for Ollama: {model_name}")

        # Update the config with possibly adjusted name
        config['model']['name'] = model_name

        # Write the updated config back
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        # Verify model is available in Ollama
        result = subprocess.run(["ollama", "list"], stdout=subprocess.PIPE, text=True)
        models_output = result.stdout
        logger.info(f"Available models:\n{models_output}")

        # Run the model playground with the updated config
        cmd = ["python", "open_source_examples/model_playground.py", DATA_PATH]
        try:
            subprocess.run(cmd, check=True)
            return True, original_model_name
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running model_playground.py: {e}")
            # Continue despite error - we might still have partial results
            return True, original_model_name
    except Exception as e:
        logger.error(f"Error in run_model_test: {e}")
        logger.error(traceback.format_exc())
        return False, model_name

def run_conversation_evaluation():
    """Run conversation clustering evaluation."""
    logger.info("Running conversation clustering evaluation...")
    try:
        cmd = ["python", "conversation_metrics.py", f"data/groups/{COMMUNITY}"]
        subprocess.run(cmd, check=True)

        # Find the most recent metrics file
        metrics_files = [f for f in os.listdir(f"data/groups/{COMMUNITY}")
                        if f.startswith(f"metrics_conversations_{COMMUNITY}")]

        if not metrics_files:
            logger.error("No conversation metrics file found")
            return None

        latest_file = max(metrics_files,
                         key=lambda x: os.path.getmtime(os.path.join(f"data/groups/{COMMUNITY}", x)))
        metrics_path = os.path.join(f"data/groups/{COMMUNITY}", latest_file)

        logger.info(f"Using conversation metrics from: {latest_file}")
        return pd.read_csv(metrics_path)
    except Exception as e:
        logger.error(f"Error running conversation evaluation: {e}")
        logger.error(traceback.format_exc())
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

        logger.info(f"Using spam metrics from: {latest_file}")
        return pd.read_csv(metrics_path)
    except Exception as e:
        logger.error(f"Error running spam evaluation: {e}")
        logger.error(traceback.format_exc())
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

        logger.info(f"Using topic metrics from: {latest_file}")
        return pd.read_csv(metrics_path)
    except Exception as e:
        logger.error(f"Error running topic evaluation: {e}")
        logger.error(traceback.format_exc())
        return None

def combine_results(model_name, conversation_df, spam_df, topic_df):
    """Combine results from all evaluations for a model."""
    results = {"model": model_name}

    # Get timestamp from current run for matching
    current_timestamp = datetime.now().strftime('%H%M%S')
    today_date = datetime.now().strftime('%Y%m%d')

    # Clean model name for matching - try all possible formats
    base_model_name = model_name.split(':')[0] if ':' in model_name else model_name
    alt_model_name = model_name.replace(':', '-')  # Format with hyphen
    all_name_patterns = [
        model_name, base_model_name, alt_model_name,
        model_name.lower(), base_model_name.lower(), alt_model_name.lower()
    ]

    # Look for recent files in the data directory to extract timestamp
    data_dir = f"data/groups/{COMMUNITY}"
    try:
        recent_labels = [
            f for f in os.listdir(data_dir)
            if f.startswith(f"labels_{today_date}") and
               any(pattern in f for pattern in all_name_patterns)
        ]

        if recent_labels:
            # Extract the timestamp from the most recent file
            try:
                most_recent = max(recent_labels, key=lambda x: os.path.getmtime(os.path.join(data_dir, x)))
                timestamp_match = re.search(r'_(\d{8})_(\d{6})_', most_recent)
                if timestamp_match:
                    current_timestamp = timestamp_match.group(2)
                    logger.info(f"Found timestamp {current_timestamp} for matching {model_name}")
            except Exception as e:
                logger.warning(f"Couldn't extract timestamp from recent files: {e}")
    except Exception as e:
        logger.warning(f"Error scanning for recent label files: {e}")

    # Add timestamp to name patterns for matching
    all_name_patterns.append(current_timestamp)

    # Log what we're looking for
    logger.info(f"Searching for metrics using patterns: {all_name_patterns}")

    # Extract conversation metrics with enhanced matching
    if conversation_df is not None:
        try:
            logger.info(f"Conversation DataFrame columns: {conversation_df.columns.tolist()}")

            model_conv_results = None

            # 1. Try string match on model column if it exists
            if 'model' in conversation_df.columns and pd.api.types.is_string_dtype(conversation_df['model']):
                for name_pattern in all_name_patterns:
                    matches = conversation_df[conversation_df['model'].str.contains(str(name_pattern), case=False, na=False, regex=False)]
                    if not matches.empty:
                        model_conv_results = matches
                        logger.info(f"Matched on model column with pattern: {name_pattern}")
                        break

            # 2. Try exact match on numeric model column (might be timestamp)
            if model_conv_results is None and 'model' in conversation_df.columns and pd.api.types.is_numeric_dtype(conversation_df['model']):
                for name_pattern in all_name_patterns:
                    if str(name_pattern).isdigit():
                        matches = conversation_df[conversation_df['model'] == int(name_pattern)]
                        if not matches.empty:
                            model_conv_results = matches
                            logger.info(f"Matched on numeric model column with value: {name_pattern}")
                            break

            # 3. Try matching on label_file column
            if model_conv_results is None:
                for col in conversation_df.columns:
                    if 'file' in col.lower() and pd.api.types.is_string_dtype(conversation_df[col]):
                        for name_pattern in all_name_patterns:
                            matches = conversation_df[conversation_df[col].str.contains(str(name_pattern), case=False, na=False, regex=False)]
                            if not matches.empty:
                                model_conv_results = matches
                                logger.info(f"Matched on {col} column with pattern: {name_pattern}")
                                break

            # Extract metrics if we found a match
            if model_conv_results is not None and not model_conv_results.empty:
                if 'ari' in model_conv_results.columns:
                    results["ari_score"] = float(model_conv_results['ari'].iloc[0])
                    logger.info(f"Found ARI score: {results['ari_score']}")
                if 'n_messages' in model_conv_results.columns:
                    results["messages_processed"] = int(model_conv_results['n_messages'].iloc[0])
            else:
                logger.warning(f"No conversation metrics found for model: {model_name}")
        except Exception as e:
            logger.warning(f"Error extracting conversation metrics: {e}")
            logger.warning(traceback.format_exc())

    # Extract spam detection metrics
    if spam_df is not None:
        try:
            logger.info(f"Spam DataFrame columns: {spam_df.columns.tolist()}")

            model_spam_results = None

            # Try same matching strategies as above
            # 1. String model column
            if 'model' in spam_df.columns and pd.api.types.is_string_dtype(spam_df['model']):
                for name_pattern in all_name_patterns:
                    matches = spam_df[spam_df['model'].str.contains(str(name_pattern), case=False, na=False, regex=False)]
                    if not matches.empty:
                        model_spam_results = matches
                        break

            # 2. Numeric model column
            if model_spam_results is None and 'model' in spam_df.columns and pd.api.types.is_numeric_dtype(spam_df['model']):
                for name_pattern in all_name_patterns:
                    if str(name_pattern).isdigit():
                        matches = spam_df[spam_df['model'] == int(name_pattern)]
                        if not matches.empty:
                            model_spam_results = matches
                            break

            # 3. Label file column
            if model_spam_results is None:
                for col in spam_df.columns:
                    if 'file' in col.lower() and pd.api.types.is_string_dtype(spam_df[col]):
                        for name_pattern in all_name_patterns:
                            matches = spam_df[spam_df[col].str.contains(str(name_pattern), case=False, na=False, regex=False)]
                            if not matches.empty:
                                model_spam_results = matches
                                break

            # Extract metrics
            if model_spam_results is not None and not model_spam_results.empty:
                metrics_to_extract = ['accuracy', 'precision', 'recall', 'f1']
                for metric in metrics_to_extract:
                    if metric in model_spam_results.columns:
                        # Handle NaN values
                        value = model_spam_results[metric].iloc[0]
                        if pd.notnull(value):
                            results[f"spam_{metric}"] = float(value)
        except Exception as e:
            logger.warning(f"Error extracting spam metrics: {e}")
            logger.warning(traceback.format_exc())

    # Extract topic evaluation metrics
    if topic_df is not None:
        try:
            logger.info(f"Topic DataFrame columns: {topic_df.columns.tolist()}")

            # Find AVERAGE scores across all topics
            model_topic_results = None

            # First try to find the model by name
            if 'model' in topic_df.columns and pd.api.types.is_string_dtype(topic_df['model']):
                for name_pattern in all_name_patterns:
                    matches = topic_df[
                        (topic_df['model'].str.contains(str(name_pattern), case=False, na=False, regex=False)) &
                        (topic_df['topic'] == 'AVERAGE')
                    ]
                    if not matches.empty:
                        model_topic_results = matches
                        break

            # Extract topic metrics
            if model_topic_results is not None and not model_topic_results.empty:
                metrics_to_extract = ['information_density', 'redundancy', 'relevance', 'efficiency', 'overall']
                for metric in metrics_to_extract:
                    if metric in model_topic_results.columns:
                        value = model_topic_results[metric].iloc[0]
                        if pd.notnull(value):
                            results[f"topic_{metric}"] = float(value)
        except Exception as e:
            logger.warning(f"Error extracting topic metrics: {e}")
            logger.warning(traceback.format_exc())

    # Log the results we found
    logger.info(f"Combined results for {model_name}: {results}")
    return results

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Evaluate multiple models on conversation analysis tasks")
    parser.add_argument("--models", nargs='+', help="Specific models to test")
    parser.add_argument("--primary-only", action="store_true", help="Only test primary recommended models")
    parser.add_argument("--alternative-only", action="store_true", help="Only test alternative models")
    parser.add_argument("--debug", action="store_true", help="Print more diagnostic information")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

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

        try:
            if pull_model(model):
                success, model_name_used = run_model_test(model)
                if success:
                    # Run all three evaluations
                    conversation_results = run_conversation_evaluation()
                    if args.debug and conversation_results is not None:
                        logger.debug(f"Conversation results sample:\n{conversation_results.head()}")
                    time.sleep(1)  # Brief pause

                    spam_results = run_spam_evaluation()
                    if args.debug and spam_results is not None:
                        logger.debug(f"Spam results sample:\n{spam_results.head()}")
                    time.sleep(1)

                    topic_results = run_topic_evaluation()
                    if args.debug and topic_results is not None:
                        logger.debug(f"Topic results sample:\n{topic_results.head()}")

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
        except Exception as e:
            logger.error(f"Error testing model {model}: {e}")
            logger.error(traceback.format_exc())

        time.sleep(5)  # Pause between models

    try:
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
                # Check if we have any actual ARI scores
                if final_df['ari_score'].notna().any():
                    sorted_results = final_df.sort_values(by='ari_score', ascending=False)
                    summary_columns = ['model', 'ari_score', 'messages_processed', 'category']
                    columns_to_show = [col for col in summary_columns if col in sorted_results.columns]
                    logger.info(sorted_results[columns_to_show])

                    # Find the best model
                    best_model = sorted_results.iloc[0]
                    logger.info(f"\n✅ BEST MODEL: {best_model['model']} with ARI score of {best_model['ari_score']:.3f}")
                    logger.info(f"   Improvement over baseline: {(best_model['ari_score'] - 0.219):.3f} points")
                else:
                    logger.info("No valid ARI scores found in results")
            else:
                logger.info("No ARI scores column in results")
        else:
            logger.error("No results were collected")
    except Exception as e:
        logger.error(f"Error in summary generation: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
