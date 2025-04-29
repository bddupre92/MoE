# FuseMoE Integration and Visualization Report

## Introduction

This report details the process of integrating the FuseMoE library (https://github.com/aaronhan223/FuseMoE) into our existing migraine prediction pipeline. The primary goal was to leverage FuseMoE's advanced Mixture-of-Experts (MoE) capabilities to potentially enhance model performance and provide deeper insights into expert contributions. The integration involved cloning the FuseMoE repository, setting up dependencies, creating an integration module (`fusemoe_integration.py`), and developing an interactive Jupyter notebook (`fusemoe_visualization.ipynb`) for data generation, model training, and visualization.

## Integration Process

The integration process involved several key steps:

1.  **Repository Cloning:** The FuseMoE repository was successfully cloned into the `/home/ubuntu/FuseMoE/` directory.
2.  **Dependency Management:** Initial attempts to install dependencies using the provided `requirements.txt` failed due to conflicts between the specified older versions (e.g., PyTorch 1.7.1) and our current environment (Python 3.10, PyTorch 2.7.0). The requirements were modified to install compatible versions, which required additional compatibility fixes later in the process.
3.  **Integration Module (`fusemoe_integration.py`):** A dedicated Python module was created to bridge our existing pipeline with FuseMoE. This module included:
    *   An `FuseMoEAdapter` class to handle the creation of FuseMoE components (`MoE` and `HierarchicalMoE`).
    *   A `FuseMoEMigraineModel` class wrapping the FuseMoE components into a structure compatible with our training pipeline.
    *   Robust fallback mechanisms to revert to a simpler, non-MoE implementation if FuseMoE components failed during initialization or execution.
    *   Data adaptation logic (`adapt_data_for_fusemoe`) to ensure compatibility between our data format and FuseMoE's expected inputs.
4.  **Configuration (`MoEConfig`):** The integration code was updated to correctly instantiate and utilize the `MoEConfig` class required by FuseMoE models, mapping parameters from our project's configuration.
5.  **Import Corrections:** Class name mismatches between our integration code and the FuseMoE library were identified and corrected (e.g., `HierarchicalMoE_seq` vs. `HierarchicalMoE`, `SparseMoE` vs. `MoE`).
6.  **Jupyter Notebook (`fusemoe_visualization.ipynb`):** An interactive notebook was developed using `ipywidgets` and `plotly` to visualize the data generation process, train the integrated model, evaluate performance, and explore expert contributions.

## Challenges Encountered and Solutions Implemented

Several significant challenges were encountered during the integration, primarily stemming from library version incompatibilities. Through systematic debugging and code modifications, all issues were successfully resolved:

1.  **Dependency Conflicts:** FuseMoE was designed for older libraries (specifically PyTorch 1.7.1). Running it with our newer environment (PyTorch 2.7.0) required modifying dependencies and implementing compatibility fixes in the FuseMoE code.

2.  **Import Errors:** Initial integration attempts failed due to incorrect class names being imported from the FuseMoE library. These were identified and corrected by updating import statements (e.g., `from core.sparse_moe import MoE as SparseMoE`).

3.  **Configuration Mismatches:** The FuseMoE models required specific configuration objects (`MoEConfig`) and parameter structures. We fixed these by:
    * Ensuring `num_experts` and `top_k` were passed as lists for `HierarchicalMoE`
    * Changing the `gating` parameter from a single string to a list `["softmax", "softmax"]` for hierarchical levels
    * Setting `router_type` to "joint" instead of "linear" to avoid modality iteration issues

4.  **Tensor Size Mismatch in SparseMoE:** We encountered a runtime error: `inconsistent tensor size, expected tensor [42] and src [4] to have the same number of elements`. This was fixed by changing the `router_type` to "joint" in the `MoEConfig` for `SparseMoE` layers, which prevented the code from incorrectly iterating through modalities.

5.  **Clean Logits Reference Error in HierarchicalMoE:** We encountered an error `UnboundLocalError: local variable 'clean_logits' referenced before assignment` in the `_get_logits` method. This was fixed by:
    * Modifying the `HierarchicalMoE.__init__` method to properly initialize `self.w_gate` and `self.w_noise` as persistent `nn.ParameterList` attributes
    * Updating the `_get_logits` method to use these class attributes instead of creating new parameters on each call
    * Adding a default initialization for `clean_logits` and proper error handling for invalid gating types

6.  **Batch Dimension Mismatch in SparseDispatcher:** The most challenging issue was a size mismatch error: `Size does not match at dimension 0 expected index [6, 1] to be no larger than self [5, 4]`. This was fixed by:
    * Replacing the complex `SparseDispatcher.__init__` implementation in `hme_seq.py` with the simpler logic from `sparse_moe.py`
    * Aligning the calculation of `_expert_index` and `_batch_index` to ensure compatible dimensions
    * Removing the problematic `torch.unique_consecutive` call that was causing the batch dimension mismatch

All these fixes were systematically implemented and tested, resulting in a fully functional integration of the FuseMoE components with our PyTorch 2.7.0 environment.

## Visualization Results (Authentic FuseMoE Implementation)

With all integration issues resolved, the Jupyter notebook can now be executed using the actual FuseMoE components rather than the fallback implementation. The generated interactive visualizations include:

1.  **Data Generation:** Interactive plots (heatmaps, bar charts, radar charts, line charts) visualizing the synthetic data generated for sleep, weather, stress/diet, and physiological inputs, along with the target variable distribution.

2.  **Model Training:** Interactive plots displaying the training and validation loss curves, as well as validation accuracy over the training epochs for the actual FuseMoE model with its hierarchical mixture-of-experts architecture.

3.  **Performance Evaluation:** Interactive visualizations of the FuseMoE model's performance on the test set, including a confusion matrix, ROC curve, precision-recall curve, and a summary bar chart of key metrics (accuracy, precision, recall, F1 score, ROC AUC).

4.  **Expert Contributions:** Most importantly, the notebook now provides authentic visualizations of expert contributions from the actual FuseMoE gating mechanisms, including:
    * Bar charts showing the relative importance of each expert across different input modalities
    * Heatmaps displaying the expert activation patterns across the test dataset
    * Pie charts illustrating the overall distribution of expert utilization
    * Network diagrams showing the hierarchical structure of expert routing

These visualizations provide valuable insights into how the mixture-of-experts model distributes tasks among specialists and how the gating network routes different input patterns to the most appropriate experts.

## Conclusion and Next Steps

The integration of FuseMoE presented significant challenges due to library version incompatibilities, but through systematic debugging and targeted code modifications, all issues were successfully resolved. The FuseMoE components now execute correctly within our PyTorch 2.7.0 environment, allowing for authentic mixture-of-experts modeling and visualization.

The resulting notebook provides valuable interactive tools for exploring the data, training process, model performance, and—most importantly—the expert contribution patterns that are the hallmark of mixture-of-experts architectures.

Potential next steps include:

1.  **Optimize Expert Configuration:** Experiment with different numbers of experts and routing configurations to potentially improve model performance.

2.  **Explore Interpretability:** Leverage the expert contribution visualizations to gain deeper insights into which patterns in the input data trigger specific experts, potentially revealing clinically relevant subgroups within migraine triggers.

3.  **Performance Comparison:** Conduct a formal comparison between the FuseMoE architecture and traditional models to quantify the performance benefits of the mixture-of-experts approach for migraine prediction.

4.  **Deployment Preparation:** Prepare the integrated FuseMoE model for deployment in the migraine prediction application, ensuring the expert visualization components are incorporated into the dashboard.

The successful integration of FuseMoE opens up exciting possibilities for both model performance improvements and deeper insights into the complex patterns underlying migraine prediction.
