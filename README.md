# UATR-benchmark
code for "Deep Learning for Underwater Acoustic Target Recognition: A Comprehensive Review, Benchmarking and Future Directions"
 
Underwater acoustic target recognition (UATR) is a critical technology for maritime safety, maritime surveillance, and marine traffic management. Conventional approaches, such as signal processing techniques and traditional machine learning models, often exhibit limited adaptability and unreliable performance in complex underwater environments. In recent years, deep learning has emerged as the predominant methodological paradigm in this field. 

However, existing surveys rarely provide a systematic and comprehensive review of deep learningbased UATR methods. Therefore, this work aims to present a thorough review centered on deep learning for UATR, encompassing task definition, characteristics of underwater acoustic signals, key challenges, available datasets, a taxonomy of methodologies, and prospective research directions. 

Furthermore, the absence of a standardized evaluation benchmark in current UATR research hinders the reproducibility of results and complicates the fair comparison of different methods. To address this issue, we establish an open-source benchmark framework and conduct extensive experimental evaluations of representative UATR approaches. It is anticipated that this work could serve as a comprehensive reference that bridges theoretical research and practical application, thereby facilitating further advancement in the field of UATR.

In summary, our work makes the following three key contributions:

• A Comprehensive Review and Taxonomy: 

We conduct a systematic review of existing deep learning-based UATR methods and classify them into three categories: waveform-based methods, spectrogram-based methods, and multimodal data-based methods.

• An Integrated Discussion of Core Aspects: 

We provide an in-depth discussion of the fundamental principles of underwater acoustic signals, the challenges inherent in the UATR task, currently available datasets, and promising future research directions.

• A Standardized Benchmark with Practical Data Partitioning: 

We reproduce a suite of representative UATR methods to establish a unified benchmark for comprehensive evaluation and comparison. This benchmark incorporates optimized data partitioning strategies to ensure that recognition performance reflects real-world application scenarios.

**For using:**

We provide dataset loading routines, feature extraction pipelines, and model architecture implementations for each method in their respective directories. In our implementation, dataset.py returns the tuple (WAV audio file path, label) to avoid input data conflicts across different algorithms. Consequently, an additional custom collate_fn is required, which introduces extra I/O overhead. Users may customize the data loader return values as needed—for instance, directly returning (audio feature tensor, label). Please follow the configurations specified below for the concrete training workflow:

For fair comparison across all involved UATR models, we adopt identical audio preprocessing pipelines, consistent training configurations, and unified experimental settings throughout all experiments. All models follow the same resampling, segmentation and overlap strategies, and are trained with the same optimizer, learning rate scheduler, maximum training epochs, learning rate decay strategy, early stopping criterion, as well as multiple initial learning rate training schemes. Only the batch size is set separately according to the scale of each dataset, while keeping it fixed for all models on the same dataset. Such a consistent experimental protocol guarantees the fairness and credibility of all comparative results.
To be specific, we use the Adam optimizer and the StepLR learning rate scheduler for training. All model methods are trained for 100 epochs, and the learning rate decayed to 0.4 of the original one every 20 epochs. Early stop is applied if the validation accuracy does not improve for 20 consecutive epochs. The batch size is set to 16 for ShipsEar, and 32 for DeepShip and Oceanship. During model training, we first compare three initial learning rates (0.001, 0.005, 0.0005) and select the optimal one based on the performance on the validation set. After all hyperparameters are finalized, we perform five independent repeated experiments under identical settings and report the average results on the unseen test set.
