"""
BioHub Cell Tracking - Embryo-Split Cross-Validation Engine

This module implements a robust cross-validation scoring framework to prevent
leaderboard overfitting by ensuring no single embryo crosses train-validation splits.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict


class EmbryoSplitEvaluator:
    """
    Cross-validation evaluator that splits data by embryo_id to prevent data leakage.
    
    Folder structure follows: '{embryo_id}_{field_of_view}'
    This ensures no single embryo appears in both training and validation sets.
    """
    
    def __init__(self, train_dir: str = "data/train"):
        self.train_dir = train_dir
        self.embryo_datasets = self._group_datasets_by_embryo()
        
    def _group_datasets_by_embryo(self) -> Dict[str, List[str]]:
        """
        Group zarr datasets by embryo_id based on folder naming convention.
        Format: '{embryo_id}_{field_of_view}.zarr'
        """
        import os
        
        embryo_groups = defaultdict(list)
        zarr_folders = [f for f in os.listdir(self.train_dir) if f.endswith(".zarr")]
        
        for folder in zarr_folders:
            # Extract embryo_id (everything before the last underscore)
            parts = folder.replace(".zarr", "").rsplit("_", 1)
            embryo_id = parts[0] if len(parts) > 1 else folder
            embryo_groups[embryo_id].append(folder)
            
        return dict(embryo_groups)
    
    def create_splits(self, n_splits: int = 5, random_seed: int = 42) -> List[Tuple[List[str], List[str]]]:
        """
        Create k-fold cross-validation splits ensuring embryo-level separation.
        
        Args:
            n_splits: Number of CV folds
            random_seed: Random seed for reproducibility
            
        Returns:
            List of (train_datasets, val_datasets) tuples
        """
        np.random.seed(random_seed)
        embryo_ids = list(self.embryo_datasets.keys())
        np.random.shuffle(embryo_ids)
        
        splits = []
        fold_size = len(embryo_ids) // n_splits
        
        for fold in range(n_splits):
            val_embryos = embryo_ids[fold * fold_size : (fold + 1) * fold_size]
            train_embryos = [e for e in embryo_ids if e not in val_embryos]
            
            val_datasets = []
            for embryo in val_embryos:
                val_datasets.extend(self.embryo_datasets[embryo])
                
            train_datasets = []
            for embryo in train_embryos:
                train_datasets.extend(self.embryo_datasets[embryo])
            
            splits.append((train_datasets, val_datasets))
            
        return splits
    
    @staticmethod
    def compute_edge_jaccard(pred_edges: set, true_edges: set, tolerance: float = 2.0) -> float:
        """
        Compute Jaccard index for edge matching with spatial tolerance.
        
        Args:
            pred_edges: Set of predicted edges (source_id, target_id)
            true_edges: Set of ground truth edges (source_id, target_id)
            tolerance: Spatial tolerance in microns for edge matching
            
        Returns:
            Jaccard index between predicted and true edges
        """
        if len(true_edges) == 0:
            return 1.0 if len(pred_edges) == 0 else 0.0
        
        # Direct matches
        direct_matches = pred_edges & true_edges
        
        # For tolerance-based matching (would need coordinate info in full implementation)
        # This is a simplified version
        jaccard = len(direct_matches) / len(pred_edges | true_edges)
        
        return jaccard
    
    @staticmethod
    def detect_divisions_from_edges(edges: np.ndarray, nodes: np.ndarray) -> set:
        """
        Detect cell division events from tracking results.
        A division occurs when one parent cell links to two or more daughter cells.
        
        Args:
            edges: Array of edge connections [source_id, target_id]
            nodes: Array of node coordinates [t, z, y, x]
            
        Returns:
            Set of division events (parent_node_id)
        """
        if len(edges) == 0:
            return set()
        
        # Count outgoing edges per node
        from collections import defaultdict
        outgoing_count = defaultdict(int)
        
        for source_id, target_id in edges:
            outgoing_count[source_id] += 1
        
        # Divisions are nodes with 2+ outgoing edges (parent splits into daughters)
        divisions = {node_id for node_id, count in outgoing_count.items() if count >= 2}
        
        return divisions
    
    @staticmethod
    def extract_divisions_from_graph(edges: set, nodes: np.ndarray) -> set:
        """
        Extract division events from edge set for evaluation.
        
        Args:
            edges: Set of edge tuples (source_id, target_id)
            nodes: Node coordinates for temporal validation
            
        Returns:
            Set of division parent IDs
        """
        if not edges:
            return set()
        
        # Build adjacency to find nodes with multiple children
        children_count = defaultdict(int)
        for source, target in edges:
            children_count[source] += 1
        
        # Division = parent with 2+ children
        divisions = {node_id for node_id, count in children_count.items() if count >= 2}
        
        return divisions
    
    @staticmethod
    def compute_division_jaccard(pred_divisions: set, true_divisions: set) -> float:
        """
        Compute Jaccard index for division events (cell splits).
        
        Args:
            pred_divisions: Set of predicted division events
            true_divisions: Set of ground truth division events
            
        Returns:
            Jaccard index for division detection
        """
        if len(true_divisions) == 0:
            return 1.0 if len(pred_divisions) == 0 else 0.0
        
        matches = pred_divisions & true_divisions
        jaccard = len(matches) / len(pred_divisions | true_divisions)
        
        return jaccard
    
    @staticmethod
    def compute_competition_score(pred_nodes: np.ndarray, true_nodes: np.ndarray,
                                  pred_edges: set, true_edges: set,
                                  pred_divisions: set, true_divisions: set,
                                  edge_tolerance: float = 2.0) -> float:
        """
        Compute the exact competition evaluation score:
        score = adjusted_edge_jaccard + 0.1 * division_jaccard
        
        Args:
            pred_nodes: Predicted node coordinates [t, z, y, x]
            true_nodes: Ground truth node coordinates [t, z, y, x]
            pred_edges: Set of predicted edges
            true_edges: Set of ground truth edges
            pred_divisions: Set of predicted divisions
            true_divisions: Set of ground truth divisions
            edge_tolerance: Spatial tolerance for edge matching
            
        Returns:
            Competition score
        """
        # Compute edge Jaccard with tolerance
        edge_jaccard = EmbryoSplitEvaluator.compute_edge_jaccard(
            pred_edges, true_edges, edge_tolerance
        )
        
        # Compute division Jaccard
        division_jaccard = EmbryoSplitEvaluator.compute_division_jaccard(
            pred_divisions, true_divisions
        )
        
        # Competition formula
        score = edge_jaccard + 0.1 * division_jaccard
        
        return score
    
    def evaluate_submission(self, submission_path: str, ground_truth_path: str) -> Dict[str, float]:
        """
        Evaluate a submission file against ground truth.
        
        Args:
            submission_path: Path to submission CSV
            ground_truth_path: Path to ground truth CSV
            
        Returns:
            Dictionary of evaluation metrics
        """
        sub_df = pd.read_csv(submission_path)
        gt_df = pd.read_csv(ground_truth_path)
        
        # Extract nodes and edges
        pred_nodes = sub_df[sub_df['row_type'] == 'node'][['t', 'z', 'y', 'x']].values
        true_nodes = gt_df[gt_df['row_type'] == 'node'][['t', 'z', 'y', 'x']].values
        
        pred_edges = set(tuple(row) for row in sub_df[sub_df['row_type'] == 'edge'][['source_id', 'target_id']].values)
        true_edges = set(tuple(row) for row in gt_df[gt_df['row_type'] == 'edge'][['source_id', 'target_id']].values)
        
        # Divisions (edges where one node has multiple children - simplified)
        pred_divisions = set()  # Would need lineage analysis
        true_divisions = set()
        
        # Compute metrics
        edge_jaccard = self.compute_edge_jaccard(pred_edges, true_edges)
        division_jaccard = self.compute_division_jaccard(pred_divisions, true_divisions)
        competition_score = edge_jaccard + 0.1 * division_jaccard
        
        return {
            'edge_jaccard': edge_jaccard,
            'division_jaccard': division_jaccard,
            'competition_score': competition_score,
            'num_pred_nodes': len(pred_nodes),
            'num_true_nodes': len(true_nodes),
            'num_pred_edges': len(pred_edges),
            'num_true_edges': len(true_edges)
        }


def run_unit_tests():
    """Run comprehensive unit tests for evaluation metrics."""
    print("\n=== RUNNING UNIT TESTS ===\n")
    
    evaluator = EmbryoSplitEvaluator()
    
    # Test 1: Perfect edge matching
    print("Test 1: Perfect edge matching")
    pred_edges = {(1, 2), (2, 3), (3, 4)}
    true_edges = {(1, 2), (2, 3), (3, 4)}
    jaccard = evaluator.compute_edge_jaccard(pred_edges, true_edges)
    assert abs(jaccard - 1.0) < 1e-6, f"Expected 1.0, got {jaccard}"
    print(f"  ✓ Perfect match Jaccard: {jaccard:.4f}")
    
    # Test 2: No edge matching
    print("\nTest 2: No edge matching")
    pred_edges = {(1, 2), (3, 4)}
    true_edges = {(5, 6), (7, 8)}
    jaccard = evaluator.compute_edge_jaccard(pred_edges, true_edges)
    assert abs(jaccard - 0.0) < 1e-6, f"Expected 0.0, got {jaccard}"
    print(f"  ✓ No match Jaccard: {jaccard:.4f}")
    
    # Test 3: Partial edge matching
    print("\nTest 3: Partial edge matching")
    pred_edges = {(1, 2), (2, 3), (3, 4), (4, 5)}
    true_edges = {(1, 2), (2, 3), (5, 6)}
    jaccard = evaluator.compute_edge_jaccard(pred_edges, true_edges)
    # Union = {(1,2), (2,3), (3,4), (4,5), (5,6)} = 5, Intersection = {(1,2), (2,3)} = 2
    expected = 2.0 / 5.0
    assert abs(jaccard - expected) < 1e-6, f"Expected {expected}, got {jaccard}"
    print(f"  ✓ Partial match Jaccard: {jaccard:.4f} (expected: {expected:.4f})")
    
    # Test 4: Empty ground truth
    print("\nTest 4: Empty ground truth edges")
    pred_edges = {(1, 2)}
    true_edges = set()
    jaccard = evaluator.compute_edge_jaccard(pred_edges, true_edges)
    assert abs(jaccard - 0.0) < 1e-6, f"Expected 0.0, got {jaccard}"
    print(f"  ✓ Empty GT Jaccard: {jaccard:.4f}")
    
    # Test 5: Empty prediction
    print("\nTest 5: Empty prediction edges")
    pred_edges = set()
    true_edges = {(1, 2)}
    jaccard = evaluator.compute_edge_jaccard(pred_edges, true_edges)
    assert abs(jaccard - 0.0) < 1e-6, f"Expected 0.0, got {jaccard}"
    print(f"  ✓ Empty pred Jaccard: {jaccard:.4f}")
    
    # Test 6: Division Jaccard
    print("\nTest 6: Division Jaccard")
    pred_divs = {1, 2}
    true_divs = {1, 3}
    jaccard = evaluator.compute_division_jaccard(pred_divs, true_divs)
    # Union = {1,2,3} = 3, Intersection = {1} = 1
    expected = 1.0 / 3.0
    assert abs(jaccard - expected) < 1e-6, f"Expected {expected}, got {jaccard}"
    print(f"  ✓ Division Jaccard: {jaccard:.4f} (expected: {expected:.4f})")
    
    # Test 7: Competition score bounds
    print("\nTest 7: Competition score bounds")
    # Perfect score
    score = evaluator.compute_competition_score(
        np.array([[0, 1, 2, 3]]), np.array([[0, 1, 2, 3]]),
        {(1, 2)}, {(1, 2)}, {1}, {1}
    )
    assert score <= 1.0, f"Score should be ≤ 1.0, got {score}"
    assert score >= 0.0, f"Score should be ≥ 0.0, got {score}"
    print(f"  ✓ Perfect score: {score:.4f}")
    
    # Zero score
    score = evaluator.compute_competition_score(
        np.array([[0, 1, 2, 3]]), np.array([[0, 1, 2, 3]]),
        {(1, 2)}, {(3, 4)}, {1}, {2}
    )
    assert score >= 0.0, f"Score should be ≥ 0.0, got {score}"
    print(f"  ✓ Zero score: {score:.4f}")
    
    # Test 8: Edge permutation invariance
    print("\nTest 8: Edge permutation invariance")
    pred_edges_1 = {(1, 2), (2, 3)}
    pred_edges_2 = {(2, 3), (1, 2)}
    true_edges = {(1, 2), (2, 3)}
    jaccard_1 = evaluator.compute_edge_jaccard(pred_edges_1, true_edges)
    jaccard_2 = evaluator.compute_edge_jaccard(pred_edges_2, true_edges)
    assert abs(jaccard_1 - jaccard_2) < 1e-6, f"Jaccard should be permutation invariant"
    print(f"  ✓ Permutation invariance: {jaccard_1:.4f} == {jaccard_2:.4f}")
    
    print("\n=== ALL UNIT TESTS PASSED ===\n")


if __name__ == "__main__":
    print("--- BioHub Cross-Validation Engine Test ---")
    
    evaluator = EmbryoSplitEvaluator()
    print(f"Found {len(evaluator.embryo_groups)} embryo groups")
    
    splits = evaluator.create_splits(n_splits=5)
    print(f"Created {len(splits)} cross-validation splits")
    
    for i, (train, val) in enumerate(splits):
        print(f"Fold {i+1}: Train={len(train)}, Val={len(val)}")
    
    # Run unit tests
    run_unit_tests()
    
    # Test scoring
    pred_edges = {(1, 2), (2, 3), (3, 4)}
    true_edges = {(1, 2), (2, 3), (4, 5)}
    
    edge_jaccard = evaluator.compute_edge_jaccard(pred_edges, true_edges)
    print(f"\nEdge Jaccard: {edge_jaccard:.4f}")
    
    score = evaluator.compute_competition_score(
        np.array([[0, 1, 2, 3]]), np.array([[0, 1, 2, 3]]),
        pred_edges, true_edges, set(), set()
    )
    print(f"Competition Score: {score:.4f}")
