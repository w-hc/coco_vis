## User guide
The visualization widget targets the coco_eval object produced by the coco python api. The motivation was to carefully review the behaviors of an object detector. It may be of only limited help when it comes to collecting datasets. The user interface is largely self-explanatory. 

## Running the demo
1. Review requirements.txt
2. Put a pickled coco_eval object in this folder.
3. Follow the notebook.

### Concepts
- detections walk: object detection/instance segmentation follows the ranking based measure of Average Precision. It is natural therefore to simply go from the highest confidence detection downwards to see the correponding matching ground truth, or in case of unmatched detection, a warning message. I call this "walk", and along the way useful statistics are printed. At any given time, at most 2 objects are painted, one being the detection, the other being the matched gt if any.
- missed gts walk: undetected ground truths cannot be covered by walking over detections, and so there is a separate walk module for missed gts. This walk goes image by image, and simply display whatever missed gts there are in that image. At any given time, several gts could be displayed together.
- single image bulk: walking is a limited way to zoom in on a particular pair of dt/gt (or a few missed gts). To get a general sense of what's happening in a particular image, the user could simply switch to single image mode, where all the dts/gts are displayed together subject to a few selectable criterion. A user may in addition put in a specific image id and jump there. 
- detections walking could happen either over all images (global walk), or within a single image (local walk). The user can swap in between between these easily. The global state will be saved when you enter single image mode, in part to avoid re-computing time, and in part to make it possible to go where I left off.
- Cateogry PR curve: this displays the category level precision + confidence vs recall curve. Thanks to Ross's pull request, we can read off the scores from the coco_eval object easily.


### Limitations:
- Assume Python3
- The original motivation was to visualize bounding boxes. Gt mask is displayed properly, but dt mask is not painted. I have not tested whether this visualizer works with a coco_eval produced by a mask rcnn. This can be easily augmented. 
- Keypoint detection is not considered in this case. 
- By default, the matching threshold is configured during visualizer init to be 0.5. The area is all. All dts (at most 100 per image) will participate in matching. These are not tunable on the widget interface. But a user can create multiple widget instances with different init parameters. Not much effort on this front since I find the default pretty good.
- coco_eval calculates the IoU of a dt with crowd gt instance differently compared with a regular gt instance. Crowd IoU is calculated as intersect / dt_area, in order to be more lenient for the dt to match. I use the vanilla intersect / union throughout to get a better sense of dts on crowd.
- I might have violated a few PEP-8s due to my linter config.


### Code Design
This visualization widget makes use of ipython widget system that enables simple user interactivity in jupyter-notebook without maintaining javascript files. However, the ipython widget system has a certain inconvenicne by design, and it is reflected frequently in the code. The basic elements to build a widget are
```python
def logic(inx):
    print(inx)
wdgt = interative(
    interative_f=logic,
    inx=widgets.BoundedIntText(
        min=0, max=len(self.dtIds) - 1, step=1
    )
)

display(wdgt)
```
Hence I often make use of a factory, which looks a bit unclean, especially the closures.
```python
def typeA_wdgt(self):
    closure_var = 3
    def logic(inx):
        print(inx)
        print(closure_var)
        print(self.member1)
    interface = interative(
        interat_f=logic,
        inx=widgets.BoundedIntText(
            min=0, max=len(self.dtIds) - 1, step=1
        )
    )
    return interface

wdgt = typeA_wdgt()
display(wdgt)
```
Where the logic is too complex/verbose, I refactor it to be a separate function.


