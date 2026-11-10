Download
========

.. currentmodule:: carlanomaly

Datasets call these functions when you pass ``download=True``; use them
directly to download from your own code. :doc:`/install` lists the parts and
their sizes.

.. autofunction:: ensure_parts

.. autofunction:: part_for

.. autofunction:: carlanomaly.download.part_present

.. autofunction:: carlanomaly.download.require_parts

.. data:: PARTS

   The part names, mapped to a description of their contents. See
   :doc:`/install`.

Command line
------------

``carlanomaly-download`` calls :func:`ensure_parts`:

.. code-block:: text

   carlanomaly-download --root DIR [--parts PART ...] [--splits {train,test} ...]
                        [--no-verify] [--keep-archive] [--base-url URL]
   carlanomaly-download --list

``--parts`` defaults to ``base``, ``--splits`` to both. ``--list`` prints the
parts and their contents.
