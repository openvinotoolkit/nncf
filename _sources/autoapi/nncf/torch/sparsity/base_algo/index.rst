:orphan:

:py:mod:`nncf.torch.sparsity.base_algo`
=======================================

.. py:module:: nncf.torch.sparsity.base_algo



Classes
~~~~~~~

.. autoapisummary::

   nncf.torch.sparsity.base_algo.BaseSparsityAlgoController




.. py:class:: BaseSparsityAlgoController(target_model, sparsified_module_info)

   Bases: :py:obj:`nncf.torch.compression_method_api.PTCompressionAlgorithmController`, :py:obj:`nncf.common.sparsity.controller.SparsityController`

   Serves as a handle to the additional modules, parameters and hooks inserted
   into the original uncompressed model in order to enable algorithm-specific compression.
   Hosts entities that are to be used during the training process, such as compression scheduler and
   compression loss.

   .. py:method:: disable_scheduler()

      Disables current compression scheduler during training by changing
      it to a dummy one that does not change the compression rate.


   .. py:method:: compression_stage()

      Returns the compression stage. Should be used on saving best checkpoints
      to distinguish between uncompressed, partially compressed, and fully
      compressed models.

      :return: The compression stage of the target model.


   .. py:method:: strip_model(model, do_copy = False)

      Strips auxiliary layers that were used for the model compression, as it's
      only needed for training. The method is used before exporting the model
      in the target format.

      :param model: The compressed model.
      :param do_copy: Modify copy of the model, defaults to False.
      :return: The stripped model.



