import pandas as pd
import numpy as np
from nltk.metrics.distance import edit_distance
from tqdm import tqdm

def calculate_wer(ground_truth, prediction):
    """Calculate Word Error Rate"""
    gt_words = str(ground_truth).split()
    pred_words = str(prediction).split()
    
    distance = edit_distance(gt_words, pred_words)
    wer = distance / len(gt_words) if len(gt_words) > 0 else 1.0
    return wer

def calculate_cer(ground_truth, prediction):
    """Calculate Character Error Rate"""
    distance = edit_distance(str(ground_truth), str(prediction))
    cer = distance / len(str(ground_truth)) if len(str(ground_truth)) > 0 else 1.0
    return cer

def evaluate_predictions(df, ground_truth_col='ground_truth', prediction_col='prediction'):
    """
    Evaluate predictions using WER and CER metrics
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing ground truth and predictions
    ground_truth_col : str
        Name of the column containing ground truth text
    prediction_col : str
        Name of the column containing predicted text
    
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics and detailed results
    """
    # Calculate metrics for each sample
    results = []
    total_wer = 0
    total_cer = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Calculating metrics"):
        gt = row[ground_truth_col]
        pred = row[prediction_col]
        
        wer = calculate_wer(gt, pred)
        cer = calculate_cer(gt, pred)
        
        total_wer += wer
        total_cer += cer
        
        results.append({
            'ground_truth': gt,
            'prediction': pred,
            'wer': wer,
            'cer': cer
        })
    
    # Calculate average metrics
    avg_wer = total_wer / len(df)
    avg_cer = total_cer / len(df)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    return {
        'average_wer': avg_wer,
        'average_cer': avg_cer,
        'total_samples': len(df),
        'detailed_results': results_df
    }

