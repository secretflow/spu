.. SPU documentation master file, created by
   sphinx-quickstart on Thu Dec 31 23:12:32 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SPU
===

SPU(Secure Processing Unit) consists of a frontend **compiler** and a backend **runtime**.
The frontend compiler accepts an ML program as input and converts it into an **MPC-specific intermediate representation**.
After a series of delicate code optimizations, programs will be executed by a performant backend runtime as MPC protocols.
Based on SPU, we can run ML programs of different frameworks with minor modifications in a privacy-preserving manner.

SPU is a major part of `SecretFlow <https://www.secretflow.org.cn/docs/secretflow/en/>`_ project.
SPU is designed to be easily integrated into other distributed system, it's **recommended to use** SecretFlow to write SPU program.


Citing
------

SPU was accepted by USENIX ATC'23. Please check the paper `here <https://www.usenix.org/conference/atc23/presentation/ma>`_.

::

   @inproceedings {288747,
      author = {Junming Ma and Yancheng Zheng and Jun Feng and Derun Zhao and Haoqi Wu and Wenjing Fang and Jin Tan and Chaofan Yu and Benyu Zhang and Lei Wang},
      title = {{SecretFlow-SPU}: A Performant and {User-Friendly} Framework for {Privacy-Preserving} Machine Learning},
      booktitle = {2023 USENIX Annual Technical Conference (USENIX ATC 23)},
      year = {2023},
      isbn = {978-1-939133-35-9},
      address = {Boston, MA},
      pages = {17--33},
      url = {https://www.usenix.org/conference/atc23/presentation/ma},
      publisher = {USENIX Association},
      month = jul,
   }



.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started/index
   reference/index
   tutorials/index
   development/index
