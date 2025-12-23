# Changelog
## v0.3.0
* Improved training interface. Training config is removed. Instead, each model provides
`::trainer()` method that constructs `Model::Trainer` object, acting as a builder for setting
optional parameters.
* Added user-defined `f32` sample weights which are supplied using `with_weights` trainer method.
* More compact decision tree, the size of regressor tree is approximately `8*N`, where `N` is the
number of nodes. Classification tree requires ~ `8*N + 2*N*number_of_classes` bytes.
* Improved and restructured documentation.
* Removed adaptive class-weight packing. It is incompatible with `f32` sample weights.
* Bugfix. Fixed incorrect number of features during random forest training.

## v0.2.0
* Added classification and regression tests and a python script for comparison. This allows to
  easily verify that Rafor predicitons agrees with other ML software.
* Implemented Welford's algorithm to measure impurity score during regression tree training. This
  improves the accuracy of splitting value, resulting in smaller trees.
* Added adaptive class-weight packing during training of classification trees. During training, the
  triples `(feature_value, class, sample_weight)` are sorted for finding best splits. It is one of
  the most demanding parts of the algorithm. Since classes and sample weights are usually much smaller
  than `2^32`, Rafor checks their maximum values and may combine class and sample weight into single
  4 byte integer to sort tuples `(feature_value, packed_class_with_weight)`. This gives faster
  training of classification trees.
* Decision tree size is improved. Each node contained 2 usize pointers to node children. The
  left and right children are placed sequentially, thus a single pointer is enough. This saves 8
  bytes per node.

## v0.1.1
* Added training options: `min_samples_split` and `min_samples_leaf`.
* Classifier config classes provide public access to their data.
* Fixed config builder bug with `CommonConfigBuilder::with_max_features` -
it prevented chaining other options.
* Changed default parameter `max_features` for random forest regressor from `SQRT` to `usize::MAX`.

## v0.1.0
* Initial version.
* Implemented: random forest classifier and regressor.
