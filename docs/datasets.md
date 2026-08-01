# Dataset protocol

The study uses three task-diverse sources represented in balanced strata. The 1,200-case pool is development-only. The 900 added cases were selected before inference, group-isolated from development, checksum-frozen, and kept label-hidden through prospective controller execution. Together they form the 2,100-query evaluation scope.

The replication manifests contain three fixed-seed, stratified 210-case draws. Draws are group-disjoint, and all compared systems within a draw receive exactly the same queries. Manifests, seeds, source groups, and checksums are retained under `data/manifests/`.

Full image data are not redistributed because the source datasets retain their own licenses and access terms. Manifest rows preserve provenance and expected relative paths; users must obtain the source images from their official providers.
