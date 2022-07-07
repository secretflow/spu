Installing SPU
==============

Environment
-----------

On Linux
~~~~~~~~

SPU has been tested with the following settings:

- Anolis OS 8.4 or later
- python3.8
- 8c16g

On MacOS
~~~~~~~~

We have conducted some successful preliminary testings on macOS Monterey 12.4 with Intel processors and Apple Silicon.

However, we don't recommend you to run SPU with Apple Silicon for performance issues.

Docker Image
~~~~~~~~~~~~

Please check `official Docker image <https://github.com/secretflow/spu#docker>`_.


Binaries
--------

You could install SPU via the `official PyPI package <https://pypi.org/project/spu/>`_:

.. code-block:: bash
  
  pip install spu


From Source
-----------

If you couldn't use recommended environment above, you have to build SPU from source.

For details, please check `README.md <https://github.com/secretflow/spu#build>`_.


