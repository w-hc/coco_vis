import os.path as osp
from collections import defaultdict
from functools import partial

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

from IPython.display import display
from ipywidgets import interactive
import ipywidgets as widgets

from pycocotools import mask as maskUtils


def plot_image_with_anns(img, detections=(), gtruths=(), dpi=80):
    fig, ax = plt.subplots(1)
    fig.set_size_inches(img.shape[1] / dpi, img.shape[0] / dpi)
    ax.imshow(img)
    add_detections(detections)
    add_gtruths(gtruths, img)
    plt.show()


def add_detections(anns):
    ax = plt.gca()
    for ann in anns:
        if ann is None:
            continue
        x, y, w, h = ann['bbox']
        ax.add_patch(
            plt.Rectangle(
                (x, y), w, h,
                fill=False, edgecolor='orange', linewidth=3
            )
        )


def add_gtruths(anns, img):
    ax = plt.gca()
    # ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in anns:
        if ann is None:
            continue
        c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
        bbox = ann['bbox']
        ax.add_patch(
            plt.Rectangle(
                (bbox[0], bbox[1]), bbox[2], bbox[3],
                fill=False, facecolor=None, edgecolor=c, linewidth=2
            )
        )

        if 'segmentation' in ann:
            if type(ann['segmentation']) == list:
                # polygon
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                    polygons.append(Polygon(poly))
                    color.append(c)
            else:
                print("using crowd mask")
                height, width = img.shape[:-1]
                # mask
                if type(ann['segmentation']['counts']) == list:
                    rle = maskUtils.frPyObjects(
                        [ann['segmentation']], height, width
                    )
                else:
                    rle = [ann['segmentation']]
                m = maskUtils.decode(rle)
                img = np.ones( (m.shape[0], m.shape[1], 3) )
                if ann['iscrowd'] == 1:
                    color_mask = np.array([2.0, 166.0, 101.0]) / 255
                if ann['iscrowd'] == 0:
                    color_mask = np.random.random((1, 3)).tolist()[0]
                for i in range(3):
                    img[:, :, i] = color_mask[i]
                ax.imshow(np.dstack( (img, m * 0.5) ))

    p = PatchCollection(
        polygons, facecolor=color, linewidths=0, alpha=0.4)
    ax.add_collection(p)
    p = PatchCollection(
        polygons, facecolor='none', edgecolors=color, linewidths=2)
    ax.add_collection(p)


def process_coco_eval(coco_eval):
    """Store the eval results in a dictionary rather than an array.
    This takes minimal time for the 5000 eval set,
    and takes place only during visualizer init.
    This design may be changed if one visualizes larger datasets.
    """
    evalImgsDict = {}

    imgIds = coco_eval.params.imgIds
    catIds = coco_eval.params.catIds
    areaIds = range(len(coco_eval.params.areaRng))

    I = len(imgIds)
    A = len(areaIds)

    for k, catId in enumerate(catIds):
        for a, areaId in enumerate(areaIds):
            for i, imgId in enumerate(imgIds):
                evalImgsDict[(imgId, catId, a)] = \
                    coco_eval.evalImgs[k * A * I + a * I + i]
    coco_eval.evalImgsDict = evalImgsDict

    return coco_eval


class Visualizer():
    def __init__(
        self, coco_eval, matching_threshold=0.5, area='all',
        img_directory='/share/data/vision-greg/coco/images/val2017'
    ):
        self.coco_eval = process_coco_eval(coco_eval)
        self.img_directory = img_directory

        self.thr = coco_eval.params.iouThrs.tolist().index(matching_threshold)
        self.areaInx = self.coco_eval.params.areaRngLbl.index(area)

        self.catsName_to_jsonId = {
            '{}: {}'.format(meta['supercategory'], meta['name']): jId
            for jId, meta in coco_eval.cocoGt.cats.items()
        }
        self.jsonId_to_catsName = {
            jId: name
            for name, jId in self.catsName_to_jsonId.items()
        }
        # note that detectron json_dataset.py uses 1 based contId
        # for convenience I use 0 based.
        self.jsonId_to_contId = {
            v: i for i, v in enumerate(coco_eval.cocoGt.getCatIds())
        }
        self.contId_to_catsName = {
            contId: self.jsonId_to_catsName[jId]
            for jId, contId in self.jsonId_to_contId.items()
        }

        # global wdgts involve moderately intensive computations and are cached
        # they are deleted and swapped out on category change
        self.global_walk_dts_wdgt = None
        self.global_walk_missed_gts_wdgt = None
        # this refers to the active one of the 2 global widgets above and
        # is used to retrive the current global image ID upon switching to
        # single img wdgt. Single img wdgt picks up from where global left off
        self.global_wdgt_in_display = None

        # ---- get the order for category display.
        # Display from low mAP to high mAP
        AP_over_cls = [
            coco_eval.eval['scores'][self.thr, :, cls_inx, self.areaInx, -1].mean()
            for cls_inx, _ in self.contId_to_catsName.items()
        ]
        AP_cls_order = np.argsort(AP_over_cls)
        self.ordered_cat_name = [
            list(self.contId_to_catsName.values())[i] for i in AP_cls_order
        ]

        self.global_cat_button = self.make_category_button()

        self.make_widget = partial(
            WidgetFactory,
            coco_eval=self.coco_eval, areaInx=self.areaInx, thr=self.thr,
            img_directory=self.img_directory
        )

    def make_category_button(self):
        """simple helper that's frequently called.
        """
        return widgets.Select(
            options=self.ordered_cat_name,
            description='category', rows=15
        )

    def category_PR_curve(self):
        def logic(cat_name):
            thr = self.thr
            coco_eval = self.coco_eval
            cls_inx = self.jsonId_to_contId[ self.catsName_to_jsonId[cat_name] ]
            fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, squeeze=True)
            fig.set_figheight(4)
            fig.set_figwidth(23)
            for area_ind, area_tag in enumerate(self.coco_eval.params.areaRngLbl):
                area_ap = \
                    coco_eval.eval['precision'][thr, :, cls_inx, area_ind, -1].mean()
                axes[area_ind].plot(
                    coco_eval.eval['scores'][thr, :, cls_inx, area_ind, -1],
                    label='score')
                axes[area_ind].plot(
                    coco_eval.eval['precision'][thr, :, cls_inx, area_ind, -1],
                    label='precision')
                axes[area_ind].set_title(
                    "area {}, AP@{:.2f} {:.3f}".format(
                        area_tag, self.coco_eval.params.iouThrs[thr], area_ap)
                )
                axes[area_ind].set_xlabel("recall lvl")
                axes[area_ind].legend()
        wdgt = interactive(
            logic,
            # the category button here is sync-ed with that of the image_vis
            cat_name=self.global_cat_button
        )
        wdgt.children[-1].layout.height = '300px'
        display(wdgt)

    def image_vis(self):
        controller = self.controller_widget()
        controller.children[-1].layout.height = '1000px'
        display(controller)

    def controller_widget(self):
        mode_options = ['Global', 'Onto Single']

        def logic(mode):
            if mode == mode_options[0]:
                print("global by default is in walk mode.")
                wdgt = self.global_widgets()
                display(wdgt)
            elif mode == mode_options[1]:
                print("single supports global style walk on the "
                      "detections of this image as well as bulk view of "
                      "detections and ground truths")
                wdgt = self.single_img_widget()
                display(wdgt)
            else:
                raise ValueError("Invalid mode")
        interface = interactive(
            logic, mode=widgets.ToggleButtons(options=mode_options)
        )
        return interface

    def global_widgets(self):
        mode_options = ['dts walk (tp + fp)', 'missed gts walk']

        def logic(cat_name, mode):
            print("INFO: mark widget boundary")
            catId = self.catsName_to_jsonId[cat_name]
            # an old global wdgt is only swapped out on category change
            # this caching is necessary since global wdgt involves moderately
            # heavy computation as it ingests the full list of imgIds
            if mode == mode_options[0]:
                if self.global_walk_dts_wdgt is None or\
                        self.global_walk_dts_wdgt.catId != catId:
                    self.global_walk_dts_wdgt = self.make_widget(
                        imgIds=self.coco_eval.params.imgIds, catId=catId
                    ).walk_dts_wdgt()
                self.global_wdgt_in_display = self.global_walk_dts_wdgt
            elif mode == mode_options[1]:
                if self.global_walk_missed_gts_wdgt is None or\
                        self.global_walk_missed_gts_wdgt.catId != catId:
                    self.global_walk_missed_gts_wdgt = self.make_widget(
                        imgIds=self.coco_eval.params.imgIds, catId=catId
                    ).global_walk_missed_gts_wdgt()
                self.global_wdgt_in_display = self.global_walk_missed_gts_wdgt
            else:
                raise ValueError("Invalid mode")
            display(self.global_wdgt_in_display)

        interface = interactive(
            logic,
            mode=widgets.ToggleButtons(options=mode_options),
            cat_name=self.global_cat_button
        )
        return interface

    def single_img_widget(self):
        mode_options = ['bulk', 'walk']

        def logic(image_id, cat_name, mode):
            print("INFO: mark widget boundary")
            if image_id not in self.coco_eval.cocoGt.imgs:
                raise ValueError("{} not a valid id".format(image_id))
            catId = self.catsName_to_jsonId[cat_name]
            if mode == mode_options[0]:
                wdgt = self.make_widget(
                    imgIds=[image_id], catId=catId
                ).single_img_bulk_wdgt()
            elif mode == mode_options[1]:
                wdgt = self.make_widget(
                    imgIds=[image_id], catId=catId
                ).walk_dts_wdgt()
            display(wdgt)

        # since the wdgt is initially at global
        assert self.global_wdgt_in_display is not None
        default_id = self.global_wdgt_in_display.result
        image_id_button = widgets.IntText(value=default_id, description='img id')
        category_button = self.make_category_button()
        category_button.options = self._get_relevant_category_names(default_id)
        category_button.value = self.global_cat_button.value

        def update_cat_of_interest(*args):
            category_button.options = self._get_relevant_category_names(
                image_id_button.value
            )

        image_id_button.observe(update_cat_of_interest, 'value')

        interface = interactive(
            logic,
            image_id=image_id_button,
            cat_name=category_button,
            mode=widgets.ToggleButtons(options=mode_options)
        )
        return interface

    def _get_relevant_category_names(self, img_id):
        """
        Args:
            img_id: a scalar
        a category is relevant if there is either gts or dts in that img of
        this particular category.
        """
        cats_of_interest = []
        for name, json_id in self.catsName_to_jsonId.items():
            val = self.coco_eval.evalImgsDict.get(
                (img_id, json_id, self.areaInx), None)
            if val is not None:
                cats_of_interest.append(name)
        return tuple(cats_of_interest)


class WidgetFactory():
    def __init__(
        self, coco_eval, imgIds, catId, areaInx, thr, img_directory
    ):
        """Factory for core widgets
        Args:
            imgIds: list of coco image ids that form this information unit
                global widget supplies a list of all img ids.
                single img widget supplies a list of 1 img id
            areaInx: index into the areaRngLbls. default 0 (all)
            thr: index into the iouThrs. default 0 (0.5)
        The code is largely modelled after coco_eval.accumulate
        """
        self.img_directory = img_directory
        acc = []
        for imgId in imgIds:
            _val = coco_eval.evalImgsDict[ (imgId, catId, areaInx) ]
            if _val is not None:
                acc.append(_val)

        if len(acc) == 0:
            raise ValueError('Neither gt nor dt found')

        # ----- gts
        gtm = np.concatenate([elem['gtMatches'] for elem in acc], axis=1)
        # gt_order = np.argsort(-gtm, axis=1) # unlikely to be useful
        gtIds = np.concatenate([elem['gtIds'] for elem in acc])
        gtIgs = np.concatenate(
            [elem['gtIgnore'] for elem in acc]).astype(np.bool)
        num_valid_gts = sum(~gtIgs)

        # ----- missed gts
        # used by global walk missed
        # this handles gracefully even in the absence of matched gts
        imgId_to_unmatched_gtAnn = defaultdict(list)
        unmatched_gt_ids = gtIds[np.where(gtm[thr, :] == 0)]
        for _unmatched_gt_id in unmatched_gt_ids:
            gt = coco_eval.cocoGt.anns[_unmatched_gt_id]
            img_id = gt['image_id']
            imgId_to_unmatched_gtAnn[img_id].append(gt)

        # ----- dts
        dtscores = np.concatenate([elem['dtScores'] for elem in acc])
        order = np.argsort(-dtscores, kind='mergesort')

        sorted_dts_scores = dtscores[order]
        dtIds = np.concatenate(
            [elem['dtIds'] for elem in acc], axis=0)[order]
        dtm = np.concatenate(
            [elem['dtMatches'] for elem in acc], axis=1)[:, order]
        dtIgs = np.concatenate(
            [elem['dtIgnore'] for elem in acc], axis=1)[:, order]

        tp = np.logical_and(               dtm,  np.logical_not(dtIgs))
        fp = np.logical_and(np.logical_not(dtm), np.logical_not(dtIgs))
        tp_sum = tp.cumsum(axis=1)
        fp_sum = fp.cumsum(axis=1)
        precision = tp_sum / (fp_sum + tp_sum + np.spacing(1))
        if num_valid_gts > 0:
            recall = tp_sum / num_valid_gts
        else:
            recall = None
            print("there is no valid gts")

        self.thr = thr
        self.areaInx = areaInx
        self.catId = catId
        self.imgIds = imgIds

        self.coco_eval = coco_eval
        self.gtm = gtm
        self.gtIds = gtIds
        self.gtIgs = gtIgs
        self.num_valid_gts = num_valid_gts
        self.unmatched_gt_ids = unmatched_gt_ids
        self.imgId_to_unmatched_gtAnn = imgId_to_unmatched_gtAnn

        self.dtIds = dtIds
        self.dtm = dtm
        self.sorted_dts_scores = sorted_dts_scores
        self.dtIgs = dtIgs
        self.precision = precision
        self.recall = recall

    def read_img(self, file_name):
        abs_path = osp.join(self.img_directory, str(file_name))
        im = cv2.imread(abs_path)
        im = im[:, :, ::-1]
        return im

    def walk_dts_wdgt(self):
        thr = self.thr
        if len(self.dtIds) == 0:
            raise ValueError("NO DETECTIONS IN THIS INFORMATION UNIT")

        # ------- indexer
        walk_seq_indexer = widgets.BoundedIntText(
            min=0, max=len(self.dtIds) - 1, step=1)

        # ------- buttons start
        # the buttons control the index inside walk_seq_indexer
        # hence walk_seq_indexer is contructed above to be referred to
        def button_on_click_handler(b):
            curr_seq_inx = walk_seq_indexer.value
            dts_matching_mask = self.dtm[thr][:]
            if b.duty == 'fp':
                inds = np.where(dts_matching_mask == 0)[0]
            elif b.duty == 'tp':
                inds = np.where(dts_matching_mask > 0)[0]
            else:
                raise ValueError("invalid button duty")
            if len(inds) == 0:
                return
            # a nice circling back behavior since argmax of all false gives 0
            if b.direction == 'next':
                new_inx = inds[ np.argmax( inds > curr_seq_inx ) ]
            elif b.direction == 'prev':
                offset = np.argmax( (inds < curr_seq_inx)[::-1] )
                new_inx = inds[ len(inds) - 1 - offset ]
            walk_seq_indexer.value = new_inx

        button_array = []
        for duty in ['fp', 'tp']:
            for direction in ['next', 'prev']:
                desc = "{} {}".format(direction, duty)
                button = widgets.Button(description=desc)
                button.direction = direction
                button.duty = duty
                button.on_click(button_on_click_handler)
                button_array.append(button)
        # ------- buttons end

        def logic(seq_inx):
            # buttons cannot be made part of the interface.
            # Have to be displayed manually in the core logic itself
            ui = widgets.HBox(button_array)
            display(ui)
            curr_img_id = self._walk_dts_wdgt_logic(seq_inx)
            return curr_img_id

        interface = interactive(
            logic,
            seq_inx=walk_seq_indexer
        )
        interface.catId = self.catId
        return interface

    def _walk_dts_wdgt_logic(self, seq_inx):
        print("INFO: mark widget boundary")
        thr = self.thr
        dt_id = self.dtIds[seq_inx]
        confidence = self.sorted_dts_scores[seq_inx]
        matched_gt_id = self.dtm[thr][seq_inx]
        is_ignored = self.dtIgs[thr][seq_inx]
        curr_prec = self.precision[thr][seq_inx]
        curr_rcll = self.recall[thr][seq_inx] if self.recall is not None else 0
        max_rcll = self.recall[thr][-1] if self.recall is not None else 0
        print("walking at {}th dt out of {} dts".format(seq_inx, len(self.dtIds)))
        print("in total {} valid gts,  {} recalled".format(
            self.num_valid_gts, int(max_rcll * self.num_valid_gts)))
        print("")
        print(
            (
                "dt id: {}, gt id: {}, dt forgiven: {}\n"
                "prec so far: {:.3f}, rcll so far: {:.3f}, max rcll: {:.3f}\n"
                "confidence: {:.3f}"
            ).format(dt_id, matched_gt_id, is_ignored,
                     curr_prec, curr_rcll, max_rcll,
                     confidence)
        )
        dt = self.coco_eval.cocoDt.anns[dt_id]
        gt = self.coco_eval.cocoGt.anns.get(matched_gt_id, None)
        if gt is not None:
            assert dt['image_id'] == gt['image_id']
            iou = maskUtils.iou([dt['bbox']], [gt['bbox']], [False])
            print("box iou: {:.3f}".format(iou[0][0]))
        else:
            print("CURRENT DT UNMATCHED!")
        image_id = dt['image_id']
        print("img id: {}".format(image_id))

        im = self.read_img(self.coco_eval.cocoDt.imgs[image_id]['file_name'])
        plot_image_with_anns(im, [dt], [gt])

        return image_id

    def global_walk_missed_gts_wdgt(self):
        def logic(seq_inx):
            print("INFO: mark widget boundary")
            image_id = list(self.imgId_to_unmatched_gtAnn.keys())[seq_inx]
            im = self.read_img(self.coco_eval.cocoDt.imgs[image_id]['file_name'])
            unmatched_gts = self.imgId_to_unmatched_gtAnn[image_id]
            print("total {} unmatched objects in {} images".format(
                len(self.unmatched_gt_ids),
                len(self.imgId_to_unmatched_gtAnn.keys())
            ))
            print("image id: {}".format(image_id))
            print("{} unmatched objects in this image".format(len(unmatched_gts)))
            plot_image_with_anns(im, detections=(), gtruths=unmatched_gts)
            return image_id
        interface = interactive(
            logic,
            seq_inx=widgets.BoundedIntText(
                min=0, max=len(self.imgId_to_unmatched_gtAnn) - 1, step=1)
        )
        interface.catId = self.catId
        return interface

    def single_img_bulk_wdgt(self):
        options = [
            'matched_gts', 'unmatched_gts',
            'matched_dts', 'unmatched_dts'
        ]
        assert len(self.imgIds) == 1
        interface = interactive(
            self._single_img_bulk_wdgt_logic,
            which_to_display=widgets.SelectMultiple(
                options=options,
                value=options,
                rows=4,
            ),
            dts_score_threshold=widgets.FloatSlider(
                value=0.0, min=0, max=1.0, step=0.05,
                description='dt score threshold: ',
                continuous_update=False,
                readout=True, readout_format='.2f',
            )
        )
        return interface

    def _single_img_bulk_wdgt_logic(self, which_to_display, dts_score_threshold):
        print("INFO: mark widget boundary")
        image_id = self.imgIds[0]
        thr = self.thr

        im = self.read_img(self.coco_eval.cocoDt.imgs[image_id]['file_name'])

        def idsToAnns(split, ann_ids):
            if split == 'gts':
                src = self.coco_eval.cocoGt
            elif split == 'dts':
                src = self.coco_eval.cocoDt
            return [ src.anns[id] for id in ann_ids ]

        gts_to_show = []
        dts_to_show = []

        four_tranches = {
            "matched_gts": idsToAnns(
                'gts', self.gtIds[np.where(self.gtm[thr, :] != 0)]),
            "unmatched_gts": idsToAnns(
                'gts', self.gtIds[np.where(self.gtm[thr, :] == 0)]),
            "matched_dts": idsToAnns(
                'dts', self.dtIds[
                    np.where((self.dtm[thr, :] != 0) &
                        (self.sorted_dts_scores > dts_score_threshold))
                ]
            ),
            "unmatched_dts": idsToAnns(
                'dts', self.dtIds[
                    np.where((self.dtm[thr, :] == 0) &
                        (self.sorted_dts_scores > dts_score_threshold))
                ]
            )
        }

        for tranche_name, anns in four_tranches.items():
            print("{}: {}".format(tranche_name, len(anns)))

        for required_tranche in which_to_display:
            if required_tranche.endswith('dts'):
                dts_to_show += four_tranches[required_tranche]
            elif required_tranche.endswith('gts'):
                gts_to_show += four_tranches[required_tranche]

        plot_image_with_anns(
            im, detections=dts_to_show, gtruths=gts_to_show)
