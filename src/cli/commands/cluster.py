"""
Cluster Command - Document Clustering Analysis

Analyzes document embeddings using HDBSCAN clustering and UMAP dimensionality
reduction. Provides statistics on cluster composition and quality metrics.

Usage:
    knowbase cluster
    knowbase cluster --model BAAI/bge-large-en-v1.5 --format table
    knowbase cluster --export-umap clusters.json
    knowbase cluster --min-cluster-size 5 --format json
"""

import sys
import json
from typing import Optional, Dict, List, Any
import numpy as np
import click
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError

from src.utils.config import Config
from src.vector_store.chroma_manager import ChromaDBManager
from src.embeddings.model_loader import ModelLoader
from src.cli.utils.output import console, print_error, print_success


class ClusterCommandInput(BaseModel):
    """Validation model for cluster command inputs."""
    model: str = Field(default="BAAI/bge-large-en-v1.5", description="Embedding model")
    min_cluster_size: int = Field(default=5, ge=2, le=100, description="Minimum cluster size")
    min_samples: int = Field(default=5, ge=1, le=50, description="Minimum samples")
    export_umap: Optional[Path] = Field(None, description="Export UMAP projection")
    export_metadata: Optional[Path] = Field(None, description="Export cluster metadata")


@click.command()
@click.option(
    "-m",
    "--model",
    default="BAAI/bge-large-en-v1.5",
    help="Embedding model used for documents",
    metavar="TEXT",
)
@click.option(
    "--min-cluster-size",
    type=int,
    default=5,
    help="Minimum cluster size for HDBSCAN",
    metavar="INT",
)
@click.option(
    "--min-samples",
    type=int,
    default=5,
    help="Minimum samples for HDBSCAN",
    metavar="INT",
)
@click.option(
    "--export-umap",
    type=click.Path(),
    help="Export UMAP projection to file",
    metavar="PATH",
)
@click.option(
    "--export-metadata",
    type=click.Path(),
    help="Export cluster metadata to file",
    metavar="PATH",
)
@click.option(
    "-f",
    "--format",
    type=click.Choice(["text", "json", "table"], case_sensitive=False),
    default="text",
    help="Output format",
    metavar="TEXT",
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    help="Configuration file path",
    metavar="PATH",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable verbose output",
)
def cluster(
    model: str,
    min_cluster_size: int,
    min_samples: int,
    export_umap: Optional[str],
    export_metadata: Optional[str],
    format: str,
    config: Optional[str],
    verbose: bool,
):
    """
    Analyze document clustering in the embedding space.

    Uses HDBSCAN for clustering and UMAP for dimensionality reduction.
    Provides statistics on cluster quality and composition.

    Examples:
        knowbase cluster
        knowbase cluster --min-cluster-size 3
        knowbase cluster --export-umap clusters.json
        knowbase cluster --format json > clusters.json
    """
    try:
        # Validate inputs
        try:
            cluster_input = ClusterCommandInput(
                model=model,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                export_umap=Path(export_umap) if export_umap else None,
                export_metadata=Path(export_metadata) if export_metadata else None,
            )
        except ValidationError as e:
            print_error(f"Invalid input: {e.errors()[0]['msg']}")
            sys.exit(1)

        # Load configuration
        config_obj = Config()
        if verbose:
            console.print("[dim]Configuration loaded successfully[/dim]")

        # Initialize ChromaDB
        if verbose:
            console.print("[dim]Connecting to ChromaDB...[/dim]")

        chroma_manager = ChromaDBManager(db_path=Path(config_obj.VECTOR_DB_PATH))
        collection = chroma_manager.get_or_create_collection(
            name="subtitle_embeddings_bge_large"
        )

        # Get embeddings and metadata
        if verbose:
            console.print("[dim]Loading embeddings from ChromaDB...[/dim]")

        embeddings_data = collection.get(include=["embeddings", "metadatas"])
        embeddings = np.array(embeddings_data["embeddings"])
        metadatas = embeddings_data["metadatas"]
        ids = embeddings_data["ids"]

        if len(embeddings) == 0:
            print_error("No embeddings found in database")
            sys.exit(1)

        if verbose:
            console.print(f"[dim]Loaded {len(embeddings)} embeddings[/dim]")

        # Perform clustering
        if verbose:
            console.print("[dim]Performing clustering with HDBSCAN...[/dim]")

        try:
            import hdbscan
        except ImportError:
            print_error("hdbscan not installed. Install with: pip install hdbscan")
            sys.exit(1)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=cluster_input.min_cluster_size,
            min_samples=cluster_input.min_samples,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(embeddings)

        # Compute cluster statistics
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(labels).count(-1)

        if verbose:
            console.print(f"[dim]Found {n_clusters} clusters[/dim]")

        cluster_stats = _compute_cluster_stats(
            embeddings, labels, metadatas, ids
        )

        # Optionally compute UMAP projection
        umap_data = None
        if cluster_input.export_umap:
            if verbose:
                console.print("[dim]Computing UMAP projection...[/dim]")
            try:
                import umap
            except ImportError:
                print_error("umap not installed. Install with: pip install umap-learn")
                sys.exit(1)

            reducer = umap.UMAP(n_components=2, metric="euclidean", random_state=42)
            umap_projection = reducer.fit_transform(embeddings)
            umap_data = {
                "points": umap_projection.tolist(),
                "labels": labels.tolist(),
                "ids": ids,
            }

        # Output results
        if format.lower() == "json":
            _output_json_format(cluster_stats, umap_data)
        else:
            _output_text_format(cluster_stats, n_clusters, n_noise, verbose)

        # Export data if requested
        if cluster_input.export_umap and umap_data:
            with open(cluster_input.export_umap, "w") as f:
                json.dump(umap_data, f, indent=2)
            print_success(f"UMAP projection exported to {cluster_input.export_umap}")

        if cluster_input.export_metadata:
            with open(cluster_input.export_metadata, "w") as f:
                json.dump(cluster_stats, f, indent=2)
            print_success(f"Cluster metadata exported to {cluster_input.export_metadata}")

        if verbose:
            console.print("\n[green]✓ Clustering completed successfully[/green]")

    except KeyboardInterrupt:
        print_error("Clustering interrupted by user")
        sys.exit(130)
    except Exception as e:
        print_error(f"Error during clustering: {str(e)}")
        if verbose:
            console.print_exception()
        sys.exit(1)


def _compute_cluster_stats(
    embeddings: np.ndarray,
    labels: np.ndarray,
    metadatas: List[Dict],
    ids: List[str],
) -> Dict[str, Any]:
    """Compute statistics for each cluster."""
    stats = {
        "total_points": len(embeddings),
        "clusters": {},
    }

    for label in set(labels):
        cluster_mask = labels == label
        cluster_points = embeddings[cluster_mask]
        cluster_ids = [ids[i] for i in range(len(ids)) if cluster_mask[i]]
        cluster_meta = [metadatas[i] for i in range(len(metadatas)) if cluster_mask[i]]

        # Compute statistics
        size = np.sum(cluster_mask)
        if size > 1:
            # Centroid and average distance
            centroid = np.mean(cluster_points, axis=0)
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            avg_distance = float(np.mean(distances))
            max_distance = float(np.max(distances))
        else:
            avg_distance = 0.0
            max_distance = 0.0

        cluster_name = (
            f"Cluster {label}" if label >= 0 else "Noise Points"
        )

        stats["clusters"][cluster_name] = {
            "label": int(label),
            "size": int(size),
            "percentage": float(size / len(embeddings) * 100),
            "avg_distance_to_centroid": avg_distance,
            "max_distance_to_centroid": max_distance,
            "sample_documents": [
                {
                    "id": cluster_ids[i],
                    "filename": cluster_meta[i].get("filename", "unknown") if cluster_meta[i] else "unknown",
                }
                for i in range(min(3, len(cluster_ids)))
            ],
        }

    return stats


def _output_text_format(
    stats: Dict[str, Any], n_clusters: int, n_noise: int, verbose: bool
) -> None:
    """Output clustering results in text format."""
    console.print("\n" + "=" * 70)
    console.print(f"[bold cyan]Clustering Results[/bold cyan]\n")

    console.print(f"Total documents: {stats['total_points']}")
    console.print(f"Number of clusters: {n_clusters}")
    if n_noise > 0:
        console.print(f"Noise points: {n_noise}")

    console.print("\n[bold]Cluster Details:[/bold]")

    for cluster_name, cluster_data in sorted(
        stats["clusters"].items(),
        key=lambda x: x[1]["label"] if x[1]["label"] >= 0 else float("inf"),
    ):
        label = cluster_data["label"]
        size = cluster_data["size"]
        percentage = cluster_data["percentage"]

        console.print(f"\n  {cluster_name}:")
        console.print(f"    Size: {size} ({percentage:.1f}%)")
        console.print(
            f"    Avg distance to centroid: {cluster_data['avg_distance_to_centroid']:.4f}"
        )

        if cluster_data.get("sample_documents"):
            console.print("    Sample documents:")
            for doc in cluster_data["sample_documents"][:3]:
                filename = doc.get("filename", "unknown")
                console.print(f"      - {filename}")

    console.print("\n" + "=" * 70 + "\n")


def _output_json_format(stats: Dict[str, Any], umap_data: Optional[Dict]) -> None:
    """Output clustering results in JSON format."""
    output = {"clustering": stats}
    if umap_data:
        output["umap"] = {
            "n_points": len(umap_data["points"]),
            "exported": True,
        }
    console.print(json.dumps(output, indent=2))


if __name__ == "__main__":
    cluster()
