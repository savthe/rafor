# Changelog
## v0.1.1
* Added training options: `min_samples_split` and `min_samples_leaf`.
* Classifier config classes provide public access to their data.
* Fixed config builder bug with `CommonConfigBuilder::with_max_features` -
it prevented chaining other options.
* Changed default parameter `max_features` for random forest regressor from `SQRT` to `usize::MAX`.

## v0.1.0
* Initial version.
* Implemented: random forest classifier and regressor.
