Utilities
=========

Cache
-----

Loading pretrained weights for models via

.. code-block:: python

  create_model(..., pretrained=True)

usually requires downloading the weights from the internet. To avoid that, either to
make model creation faster or when executing code in environments without internet
access, we provide a two-part cache system.

There is a global cache directory, accessible by :py:func:`~tfimm.get_dir`. The cache
directory is determined in the following order of precedence

* Directory set by :py:func:`~tfimm.set_dir`
* Path set by env variable ``$TFIMM_HOME``
* Path ``$XDG_CACHE_HOME/tfimm`` if env variable ``$XDG_CACHE_HOME`` is set
* ``~/.cache``

Within the cache directory each model should be saved in a subdirectory with name
``model_name``. If the cache directory is ``~/.cache``, then ``resnet18`` should be
saved under ``~/.cache/resnet18``.

In some cases it might be impractical to store all models in the same directory. In
that case we can specify the cache location for individual models using
:py:func:`~tfimm.set_model_cache`. For example, if we have ``resnet18`` weights stored
at ``~/resnet18_saved``, we call

.. code-block:: python

  set_model_cache("resnet18", "~/resnet18_saved")
  # This will read weights from the above location
  model = create_model("resnet18", pretrained=True)

We can remove the cache location for a particular model by calling
:py:func:`~tfimm.clear_model_cache`. This will only stop ``tfimm`` from looking in that
directory, it will *not* delete anything from disk.

We can also list all models for which we have specified locations via
:py:func:`~tfimm.list_cached_models`. This will return a list of all models whose cache
location has been set via :py:func:`~tfimm.set_model_cache`. It will *not* look in the
general cache directory.

We can obtain the cache location for a particular model via
:py:func:`~tfimm.cached_model_path`. The cache location is determined in the following
order of precedence:

* Model-specific cache set via :py:func:`~tfimm.set_model_cache`
* General cache directory given by :py:func:`~tfimm.get_dir`


.. py:module:: tfimm

.. autofunction:: cached_model_path
.. autofunction:: clear_model_cache
.. autofunction:: get_dir
.. autofunction:: list_cached_models
.. autofunction:: set_dir
.. autofunction:: set_model_cache
