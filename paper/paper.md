---
title: 'SCALib: A Side-Channel Analysis Library'
tags:
  - Side-channel evaluation
  - Security
  - Python
  - Rust
authors:
  - name: Olivier Bronchain
    orcid: 0000-0001-7595-718X
    equal-contrib: true
    affiliation: 1
  - name: Gaëtan Cassiers
    orcid: 0000-0001-5426-9345
    equal-contrib: true
    corresponding: true
    affiliation: "1, 2, 3"
affiliations:
 - name: UCLouvain, Belgium
   index: 1
 - name: Graz University of Technology, Austria
   index: 2
 - name: Lamarr Security Research, Austria
   index: 3
date: 16 February 2023
bibliography: paper.bib

---

# Summary

Side-channel attacks exploit unintended leakage from an electronic device in
order to retrieve secret data.
In particular, attacks exploiting physical side-channels such as power
consumption or electromagnetic radiations to recover cryptographic keys are an
important threat to embedded devices.
Countermeasures against these attacks have been extensively researched for more
than two decades and are often deployed in security-critical devices.

A side-channel attack is made of three steps: first, the leakage is measured.
Then, a statistical processing is applied to this leakage in order to infer the
internal behavior of the device (typically, an intermediate state of the
cryptographic algorithm). Finally, the cryptographic key is recovered from the
known behavior [@DBLP:conf/eurocrypt/StandaertMY09].

For the statistical processing, we distinguish between two classes of attacks,
based on the use of a profiling dataset.
Such a dataset consists in leakage measurements on a device running the
cryptographic algorithm with the known key.
Profiled attacks use this data to fit a statistical model (or train a
machine-learning model) of the device, while non-profiled attacks have to rely
on *a priori* models and are therefore less powerful [@DBLP:conf/ches/ChariRR02].

There are two main approaches to evaluating security of devices against side-channel attacks.
First, attack-based evaluations try to attack the device and report their success or failure.
In case of success, the main figure of merit is the number of traces (i.e.,
number of executions of a cryptographic algorithm for which the leakage is
measured).
Second, detection-based evaluations try to detect the presence of key-dependent
leakage and sometimes quantify it.
These two types of methods can be complementary in the evaluation of a device.

Side-channel evaluations are used in various research contexts, such as
analyzing the effectiveness of a newly proposed countermeasure or analyzing a
widely deployed device.
In `SCALib`, we implement algorithms for commonly used metrics and methods in
side-channel security evaluations, attack-based and evaluation-based.
We however focus on the requirements of evaluations, and do not implement
complete attacks when they are not needed to evaluate the security of a device.

`SCALib` is distributed as a Python package and uses 16-bit integer `numpy` [@numpy] arrays
for leakage traces.
For the sake of efficiency, most algorithms are however implemented in Rust,
allowing fine control of the memory accesses and enabling efficient
parallelization.


# Statement of need

Many of the algorithms used in side-channel security evaluations are well-known
statistical techniques.
For instance, the widely used TVLA methodology is based on the Welch t-test for
the difference of means [@DBLP:conf/ches/SchneiderM15].
Also when modeling the leakage, techniques such as Linear Discriminant Analysis
(LDA) [@DBLP:conf/ches/StandaertA08] can be used.
While implementations of these algorithms are fairly easy to find, our use-case
has a few particularities that motivate dedicated implementations.
For example, the number of traces used in a evaluation can be very large,
amounting to terabytes of data, hence incremental single-pass algorithms (that
avoid the need to store and/or load multiple times the dataset) are highly
desirable.
Moreover, while the leakage samples are acquired at relatively low-resolution
(8-bit to 16-bit integers), detection of very small effect sizes is
needed, as they can potentially be exploited to mount an attack.
Besides this requirement, leakage traces contain many points (typically
thousands) and many metrics have to be computed for each of these points,
providing parallelization opportunities.
As a result of these characteristics, dedicated implementations can achieve
much better accuracy and performance than generic or naive (e.g., pure `numpy`)
ones.

On the other hand, security-specific algorithms are also used, such as key rank
estimation (which allows to know the computational cost of the last part of a
side-channel attack without actually running it) [@DBLP:conf/ches/PoussierSG16].

While there exists multiple open-source side-channel attack and evaluation
libraries, most of them offer a very limited feature set and are unmaintained.
The most comprehensive libraries are `lascar` [@lascar] and `SCAred` [@scared], which offer
implementations of some evaluation metrics and non-profiled attacks.

`SCALib` complements and improves over these libraries by providing better
implementations for the computation of two common evaluation metrics, by
providing algorithms for profiled side-channel attacks and by including a key
rank enumeration algorithm as a final evaluation step.
More precisely, for leakage metrics, we implement the Welch t-test and the
computation of the signal-to-noise ratio, and our implementations are significantly
faster than the ones of `lascar` and `SCAred` [@scabench].
Moreover, our t-test implementation includes so-called higher-order and
multivariate evaluations [@DBLP:conf/ches/SchneiderM15].
Regarding profiled attacks, `SCALib` includes an implementation of
LDA with a dimensionality reduction step (this
provides a regularization and improves classification performance) [@DBLP:conf/ches/StandaertA08].
We also implement the soft analytical side-channel attack (SASCA), which is a
variant of the belief propagation algorithm [@DBLP:conf/asiacrypt/Veyrat-CharvillonGS14].
Finally, our key-rank estimation implementation relies an efficient histogram-based algorithm [@DBLP:conf/ches/PoussierSG16].

`SCALib` has been used in many recent papers as a tool to validate new protected
designs [@DBLP:journals/tches/NagpalGPM22], to publish new attacks on public
implementations [@DBLP:journals/iacr/BronchainCS21], and also as a basis
to develop new attack and evaluation methodologies [@DBLP:journals/tches/BronchainS21].


# Acknowledgments

This work has been funded in part by the Belgian Fund for Scientific Research
(F.R.S.-FNRS) through the Equipment Project SCALAB, by the European Union (EU)
and the Walloon Region through the FEDER project USERMedia (convention number
501907-379156), and by the European Union (EU) through the ERC project 724725
(acronym SWORD).

# References
