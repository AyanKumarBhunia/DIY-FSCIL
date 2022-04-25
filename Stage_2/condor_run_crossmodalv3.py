import os
from condor import condor, Job, Configuration
from datetime import datetime

# from pinakinathc_py import SendEmail

conf = Configuration(
    universe="docker",  # OR 'vanilla'
    docker_image="registry.eps.surrey.ac.uk/deeprepo:as7",
    request_CPUs=8,
    request_memory=1024 * 32,
    request_GPUs=1,
    gpu_memory_range=[10000, 24000],
    cuda_capability=5.5,
)

with condor(
    "aisurrey-condor", project_space="ayanCV"
) as sess:  # aisurrey-condor vs condor

    # run_file = os.path.join(os.getcwd(), "train_stage2.py")
    run_file = os.path.join(os.getcwd(), "train_stage2_noval.py")
    exp_name = (
        "gann_crossmodal_grad"  # gradient surgery in stage 2 (cross modal training)
    )
    # gnn(alpha * w_avg)
    # same as above without gradient surgery
    exp_name = "gann_crossmodal_nograd"
    exp_name = "gann_crossmodal_grad_kd"
    exp_name = "crossmodal_onlykd"
    # dataloader_crossmodal, models_onlykd, no aug, no gsurgery, 6, 0.95
    exp_name = "crossmodal_onlykd_20t"  # temp = 20
    exp_name = "crossmodal_onlykd_10t_point9"
    exp_name = "crossmodal_kd_algcnn"
    exp_name = "crossmodal_kd_algcnn_3layers"
    exp_name = "crossmodal_kd_2gcnn_grad"  # models_all_three
    exp_name = "crossmodal_kd_2gcnn(0.5attdrop)_grad"  # models_all_three
    exp_name = "crossmodal_kd_algcnn_grad"  # models_algcnn_kd
    exp_name = "f_gs_gat_kd"  # dataloader_new
    exp_name = "f_gat_kd"
    exp_name = "f_kd"
    exp_name = "f_gs_kd"
    exp_name = "f_gs_gat_kd_oneshot"
    exp_name = "f_gat_kd_oneshot"
    exp_name = "f_kd_oneshot"
    exp_name = "f_gs_kd_oneshot"
    exp_name = "f_gs_gat_oneshot"
    exp_name = "f_proto_oneshot"
    exp_name = "f_gat_oneshot"
    exp_name = "f_gat"
    exp_name = "f_proto_v2"  # no weights for protos
    exp_name = "f_proto_v2_oneshot"
    exp_name = "f_gs_gat_kd_tenshot"
    exp_name = "f_gs_gat_kd_20shot"
    exp_name = "f_all_photos_gs_gat_kd_5shot"
    exp_name = "f_only_sketch"  # only sketches during 2nd stage
    exp_name = "f_4s_1p"  # one photo 4 sketches during 2nd stage
    exp_name = "f_3s_2p"  # two photo 3 sketches during 2nd stage
    exp_name = "f_2s_3p"  # two photo 3 sketches during 2nd stage
    # exp_name = 'f_ours_photo_oneshot' # photo baseline, only photos
    exp_name = "f_gs_gat_kd_15shot"
    exp_name = "f_gs_gat_kd_10way5shot"

    # exp_name = "crossmodal_kd_algcnn_surgery"

    # folder_name = '_'.join([exp_name, datetime.now().strftime("%b-%d_%H:%M:%S")])
    folder_name = exp_name
    # client.send('2ajaydas@gmail.com', folder_name + os.getcwd())

    train_nKnovel = (10,)
    train_nExemplars = (5,)
    train_nTestNovel = (train_nKnovel * 3,)  # train_nKnovel * 3
    train_nTestBase = (5 * 3,)
    test_nKnovel = (10,)
    test_nExemplars = (5,)
    test_nTestNovel = (15 * test_nKnovel,)
    test_nTestBase = (15 * 5,)

    for bs in ["_"]:  # submit a bunch of jobs

        # It will autodetect the full path of your python executable
        j = Job(
            "/vol/research/ayanCV/miniconda3/envs/ayanPY/bin/python",  # if docker, use absolute path to specify executables inside container
            run_file,
            # all arguments to the executable should be in the dictionary as follows.
            # an entry 'epochs=30' in the dict will appear as 'python <file>.py --epochs 30'
            arguments=dict(
                base_dir=os.getcwd(),
                saved_models=os.path.join(
                    os.getcwd(), f"./condor_output/{folder_name}"
                ),
                # data_dir="./datasets/cross_modal_v2",
                # root=os.environ['STORAGE'] + '/datasets/quickdraw',
                train_batch_size=1,
                train_epoch_size=600,
                test_epoch_size=600,
                train_nKnovel=train_nKnovel,
                train_nExemplars=train_nExemplars,
                train_nTestNovel=train_nTestNovel,  # train_nKnovel * 3
                train_nTestBase=train_nTestBase,
                test_nKnovel=test_nKnovel,
                test_nExemplars=test_nExemplars,
                test_nTestNovel=test_nTestNovel,
                test_nTestBase=train_nTestBase,
                # train_nExemplars =15,
                # test_nExemplars = 15
            ),
            stream_output=True,
            can_checkpoint=True,
            approx_runtime=8,  # in hours
            tag=exp_name,
            artifact_dir=f"./condor_output/{folder_name}",
        )
        # finally submit it
        sess.submit(j, conf)
