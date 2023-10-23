# SolarPanel-EL-Analyzer

## Description
Solar panels provide a cost-effective solution to powering off-grid homes and devices using a
renewable and clean source of energy. They convert sunlight into electricity by means of
photovoltaic (PV) cells. Commercial solar panels usually come with an aluminium frame and
glass lamination to protect the PV cells from environmental influences such as rain, wind, and
snow. However, these protective measures cannot always prevent damage caused by
mechanical impact from falling tree branches, hail, or from thermal stress. Damages may also
be caused by errors in the manufacturing process.

As any defects can decrease the power efficiency of solar panels, it is important to monitor
the condition of the PV cells. Visual inspection by human experts is very time-consuming, and,
apart from obvious cracks in the glass, many defects are not visible to the naked eye.
Conversely, visible imperfections do not necessarily reduce the efficiency of a solar panel.
Thus, to enable quick yet rigorous inspection of solar panels, more comprehensive scanning
and automated analysis methods are needed.

Electroluminescence (EL) imaging is a non-destructive technology that allows high-resolution
scanning of PV modules to visualize a broad range of imperfections and defects. It applies a
current to the modules which induces EL emission that can be imaged using a digital camera.
In the resulting images, defective cells appear darker than functional cells, because
disconnected areas do not irradiate. The images can subsequently be analysed automatically
using computer vision methods to detect and classify defects.
The goal of this group project is to develop and test computer vision methods for predicting
the health of PV cells in EL images of solar modules.

## Dataset
The dataset to be used in the group project is the ELPV dataset (see links and references at
the end of this document). It contains 2,624 EL images of functional and defective PV cells with
varying degrees of degradation extracted from 44 different solar modules. All cell images are
normalized with respect to size and perspective, corrected for camera lens distortions, and
manually annotated with a defect probability (a real-valued number from 0 to 1) and the type
of solar module (monocrystalline or polycrystalline).

## Task
The task is to classify cell images according to their probability of defectiveness. More
specifically, automatic classifiers should predict whether a given image contains a cell that is
fully functional (0% probability of being defective), possibly defective (33% probability), likely
defective (67% probability), or certainly defective (100% probability)

## References
ELPV Dataset. A Benchmark for Visual Identification of Defective Solar Cells in Electroluminescence Imagery.
https://github.com/zae-bayern/elpv-dataset