{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "4970d171",
   "metadata": {},
   "source": [
    "## Ceci est un Test pour Trduire ONNX to MIP"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 318,
   "id": "e18443f1",
   "metadata": {},
   "outputs": [],
   "source": [
    "from read_net_file import *\n",
    "from onnx_translator import *\n",
    "import sys\n",
    "import csv\n",
    "import torch\n",
    "\n",
    "sys.path.insert(0, '../ELINA/python_interface/')\n",
    "from optimizer import *\n",
    "from analyzer import *\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 319,
   "id": "de92a2e1",
   "metadata": {},
   "outputs": [],
   "source": [
    "net_file = \"cifar_mlp.onnx\"\n",
    "\n",
    "is_trained_with_pytorch = True\n",
    "is_gpupoly = False\n",
    "\n",
    "\n",
    "model, is_conv = read_onnx_net(net_file)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 320,
   "id": "082b52c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "#!pip install onnxruntime\n",
    "#import onnxruntime.backend as ort\n",
    "#import onnx_tf.backend as otf\n",
    "#from onnx_tf.common import data_type\n",
    "\n",
    "#import onnx\n",
    "#!pip install onnx_tf\n",
    "#from onnx_tf.backend import prepare\n",
    "## Load the ONNX model\n",
    "#onnx_model = onnx.load(net_file)\n",
    "#tf_rep = prepare(onnx_model)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 321,
   "id": "032f7bed",
   "metadata": {},
   "outputs": [],
   "source": [
    "translator = ONNXTranslator(model, True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 322,
   "id": "632d3190",
   "metadata": {},
   "outputs": [],
   "source": [
    "operations, resources = translator.translate()\n",
    "optimizer = Optimizer(operations, resources)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 323,
   "id": "b02edcb8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "image_label : \n",
      "8\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/ramzi.ben-mhenni/venvpython3.7/lib/python3.7/site-packages/ipykernel_launcher.py:21: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.\n",
      "Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations\n"
     ]
    }
   ],
   "source": [
    "\n",
    "dataset = 'cifar10'\n",
    "class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',\n",
    "               'dog', 'frog', 'horse', 'ship', 'truck']\n",
    "\n",
    "if dataset in ['mnist', 'fashion']:\n",
    "    height, width, channels = 28, 28, 1\n",
    "else:\n",
    "    height, width, channels = 32, 32, 3\n",
    "        \n",
    "filename = '../data/'+ dataset+ '_test.csv'\n",
    "csvfile = open(filename, 'r')\n",
    "tests = csv.reader(csvfile, delimiter=',')\n",
    "\n",
    "test = next(tests)\n",
    "test = next(tests)\n",
    "\n",
    "image = torch.from_numpy(np.float64(test[1:len(test)]) / np.float64(255)).reshape(1, height, width, channels).permute(0, 3, 1, 2).to('cpu')\n",
    "image_v = image.clone().permute(0, 2, 3, 1).flatten().cpu()\n",
    "\n",
    "\n",
    "label = np.int(test[0])\n",
    "# Printing output\n",
    "print(\"image_label : \" )\n",
    "print (label)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 324,
   "id": "f906a3b5",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x576 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import torchvision\n",
    "\n",
    "def imshow(img, fig):\n",
    "    img = img \n",
    "    npimg = img.numpy()\n",
    "    fig.imshow(np.transpose(npimg, (1, 2, 0)))\n",
    "    \n",
    "def plot_attack_pred(original, attack):\n",
    "    # show image and noise\n",
    "    ax, fig = plt.subplots(nrows=1,ncols=2, figsize=(10,8))\n",
    "    fig[0].set_title(\"Attack\")\n",
    "    imshow(torchvision.utils.make_grid(attack), fig[0])\n",
    "    noise = (original - attack).detach()\n",
    "    fig[1].set_title(\"Noise added to original image\")\n",
    "    imshow(10*torchvision.utils.make_grid(noise), fig[1])\n",
    "    plt.show()\n",
    "\n",
    "def plot_image_pred(img,name):\n",
    "    # show image and noise\n",
    "    ax, fig = plt.subplots(nrows=1,ncols=2, figsize=(10,8))\n",
    "    fig[0].set_title(\"Original\")\n",
    "    plt.xlabel(name)\n",
    "    imshow(torchvision.utils.make_grid(img), fig[0])\n",
    "    noise = (img - img + 1).detach()\n",
    "    fig[1].set_title(\"No Noise\")\n",
    "    imshow(10*torchvision.utils.make_grid(noise), fig[1])\n",
    "    plt.show()\n",
    "\n",
    "\n",
    "plot_image_pred(image,class_names[label])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 325,
   "id": "e08cf20a",
   "metadata": {},
   "outputs": [],
   "source": [
    "num_pixels = height * width * channels\n",
    "num_flows =  2* num_pixels\n",
    "delta = 0.02\n",
    "\n",
    "\n",
    "specLB = image.clone().permute(0, 2, 3, 1).flatten().cpu() -delta\n",
    "specUB = image.clone().permute(0, 2, 3, 1).flatten().cpu() +delta\n",
    "\n",
    "flows_LB = torch.full((num_flows,), -delta).to('cpu')\n",
    "flows_UB = torch.full((num_flows,), delta).to('cpu')\n",
    "\n",
    "speLB = torch.cat((specLB, flows_LB))\n",
    "speUB = torch.cat((specUB, flows_UB))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 326,
   "id": "0561bf94",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[('0', [1, 32, 32, 3]), ('7', [1, 150]), ('8', [1, 150]), ('9', [1, 10])]"
      ]
     },
     "execution_count": 326,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#deepzono\n",
    "# execute_list_deepzono, output_info_deepzono = optimizer.get_deepzono(nn, specLB , specUB )\n",
    "# use_dict =  optimizer.deepzono_get_dict(execute_list_deepzono)\n",
    "nn = layers()\n",
    "#deeppoly\n",
    "lexpr_weights= None\n",
    "lexpr_cst=None\n",
    "lexpr_dim=None\n",
    "uexpr_weights=None\n",
    "uexpr_cst=None\n",
    "uexpr_dim=None\n",
    "expr_size=0\n",
    "\n",
    "domain = 'deeppoly'\n",
    "timeout_lp = 100\n",
    "timeout_milp = 100\n",
    "timeout_final_lp=100\n",
    "timeout_final_milp=100\n",
    "\n",
    "use_default_heuristic = True\n",
    "output_constraints=False\n",
    "\n",
    "\n",
    "testing = False\n",
    "label= 8\n",
    "prop = 1\n",
    "spatial_constraints=None\n",
    "K=0\n",
    "s=0\n",
    "\n",
    "max_milp_neurons=10000\n",
    "approx_k=False\n",
    "\n",
    "execute_list_deeppoly, output_info_deeppoly =optimizer.get_deeppoly(nn, specLB, specUB, lexpr_weights, lexpr_cst, lexpr_dim, uexpr_weights, uexpr_cst, uexpr_dim, expr_size, spatial_constraints=None)\n",
    "\n",
    "output_info_deeppoly"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 327,
   "id": "e38f26d3",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_abstractBigM(nn,ir_list,timeout_lp,timeout_milp):\n",
    "        \"\"\"\n",
    "        processes self.ir_list and returns the resulting abstract element\n",
    "        \"\"\"\n",
    "        man = fppoly_manager_alloc()\n",
    "        element = ir_list[0].transformer(man)\n",
    "        nlb = [0]\n",
    "        nub = [0]\n",
    "        testing_nlb = []\n",
    "        testing_nub = []\n",
    "        \n",
    "        domain = 'deeppoly'\n",
    "        \n",
    "        relu_groups = False\n",
    "        use_default_heuristic = True\n",
    "        testing = True\n",
    "        \n",
    "        \n",
    "        for i in range(1, len(ir_list)):\n",
    "            if type(ir_list[i]) in [DeeppolyReluNode,DeeppolySigmoidNode,DeeppolyTanhNode,DeepzonoRelu,DeepzonoSigmoid,DeepzonoTanh]:\n",
    "                element_test_bounds = ir_list[i].transformer(nn, man, element, nlb, nub,\n",
    "                                                                  relu_groups, 'refine' in domain,\n",
    "                                                                  timeout_lp, timeout_milp,\n",
    "                                                                  use_default_heuristic, testing,\n",
    "                                                                  K=1, s=1, use_milp=True,\n",
    "                                                                  approx=0)\n",
    "                print(\"1 :\")\n",
    "                \n",
    "            else:\n",
    "                element_test_bounds = ir_list[i].transformer(nn, man, element, nlb, nub,\n",
    "                                                                  relu_groups, 'refine' in domain,\n",
    "                                                                  timeout_lp, timeout_milp,\n",
    "                                                                  use_default_heuristic, testing)\n",
    "                print(\"2 :\")\n",
    "                \n",
    "\n",
    "            if testing and isinstance(element_test_bounds, tuple):\n",
    "                element, test_lb, test_ub = element_test_bounds\n",
    "                testing_nlb.append(test_lb)\n",
    "                testing_nub.append(test_ub)\n",
    "            else:\n",
    "                element = element_test_bounds\n",
    "        if domain in [\"refinezono\", \"refinepoly\"]:\n",
    "            gc.collect()\n",
    "        if testing:\n",
    "            return element, testing_nlb, testing_nub\n",
    "        return element, nlb, nub"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 328,
   "id": "58118dad",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2 :\n",
      "1 :\n",
      "2 :\n"
     ]
    }
   ],
   "source": [
    "timeout_lp = 100\n",
    "timeout_milp = 100\n",
    "element, nlb, nub = get_abstractBigM(nn,execute_list_deeppoly,timeout_lp,timeout_milp)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 329,
   "id": "ea760eb9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "This network has 150 neurons.\n",
      "FC\n",
      "ReLU\n",
      "FC\n"
     ]
    }
   ],
   "source": [
    "print('This network has ' + str(optimizer.get_neuron_count()) + ' neurons.')\n",
    "\n",
    "for i in range(nn.numlayer):\n",
    "    print(nn.layertypes[i])\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 330,
   "id": "e950a517",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "#execute_list = execute_list_deeppoly\n",
    "#analyzer = Analyzer(execute_list, nn, domain, timeout_lp, timeout_milp, output_constraints,\n",
    "#                                use_default_heuristic, label, prop, testing, K=K, s=s,\n",
    "#                                timeout_final_lp=timeout_final_lp, timeout_final_milp=timeout_final_milp,\n",
    "#                                use_milp=use_milp, complete=complete,\n",
    "#                                partial_milp=partial_milp, max_milp_neurons=max_milp_neurons)\n",
    "#terminate_on_failure= True\n",
    "#dominant_class, nlb, nub, failed_labels, x = analyzer.analyze(terminate_on_failure=terminate_on_failure)\n",
    "\n",
    "    \n",
    "#print(failed_labels)\n",
    "#print(x)\n",
    "#print(dominant_class)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 331,
   "id": "c06173ca",
   "metadata": {},
   "outputs": [],
   "source": [
    "#milp_activation_layers = np.nonzero([l in [\"ReLU\", \"Maxpool\"] for l in nn.layertypes])[0]\n",
    "#partial_milp = 0\n",
    "#### Determine whcich layers, if any to encode with MILP\n",
    "#if partial_milp < 0:\n",
    "#    partial_milp = len(milp_activation_layers)\n",
    "#first_milp_layer = len(nn.layertypes) if partial_milp == 0 else milp_activation_layers[-min(partial_milp, len(milp_activation_layers))]\n",
    "\n",
    "#first_milp_layer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 332,
   "id": "62beaadf",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "create_model\n",
      "Set parameter FeasibilityTol to value 2.0000000000000002e-05\n",
      "handle_relu\n",
      "neurons  150\n",
      "milp_encode_idx neurons  150\n"
     ]
    }
   ],
   "source": [
    "\n",
    "relu_groups = None\n",
    "use_milp=True\n",
    "complete=True\n",
    "partial_milp=False\n",
    "is_nchw=False\n",
    "partial_milp=2\n",
    "max_milp_neurons = 1000\n",
    "\n",
    "counter, var_list, model = create_model(nn, specLB, specUB, nlb, nub, relu_groups, nn.numlayer, use_milp, is_nchw, partial_milp, max_milp_neurons)\n",
    "\n",
    "num_var = len(var_list)\n",
    "output_size = num_var - counter\n",
    "#model.write(\"model_refinepoly.lp\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 333,
   "id": "7513692b",
   "metadata": {},
   "outputs": [],
   "source": [
    "adv_label = 5\n",
    "\n",
    "#obj = LinExpr()\n",
    "#obj += 1 * var_list[counter + label]\n",
    "#obj += -1 * var_list[counter + adv_label]\n",
    "#model.setObjective(obj, GRB.MINIMIZE)\n",
    "\n",
    "#model.addConstr( var_list[counter + adv_label] >= var_list[counter + label] )\n",
    "#model.addConstr( var_list[counter + adv_label] >= var_list[counter + 0] )\n",
    "#model.addConstr( var_list[counter + adv_label] >= var_list[counter + 1] )\n",
    "#model.addConstr( var_list[counter + adv_label] >= var_list[counter + 2] )\n",
    "#model.addConstr( var_list[counter + adv_label] >= var_list[counter + 3] )\n",
    "#model.addConstr( var_list[counter + adv_label] >= var_list[counter + 4] )\n",
    "#model.addConstr( var_list[counter + adv_label] >= var_list[counter + 5] )\n",
    "#model.addConstr( var_list[counter + adv_label] >= var_list[counter + 6] )\n",
    "#model.addConstr( var_list[counter + adv_label] >= var_list[counter + 7] )\n",
    "#model.addConstr( var_list[counter + adv_label] >= var_list[counter + 8] )\n",
    "#model.addConstr( var_list[counter + adv_label] >= var_list[counter + 9] )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 334,
   "id": "0e129fa2",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "expr_qp= QuadExpr()\n",
    "\n",
    "for i in range(num_pixels):\n",
    "    expr_qp.add(var_list[i] * var_list[i] - 2*image_v[i] * var_list[i] + image_v[i]*image_v[i]) \n",
    "\n",
    "# Set objective\n",
    "model.setObjective(expr_qp, GRB.MINIMIZE)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 335,
   "id": "88466561",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Set parameter TimeLimit to value 100\n",
      "Gurobi Optimizer version 9.5.0 build v9.5.0rc5 (linux64)\n",
      "Thread count: 16 physical cores, 16 logical processors, using up to 16 threads\n",
      "Optimize a model with 511 rows, 3449 columns and 463119 nonzeros\n",
      "Model fingerprint: 0x061a703c\n",
      "Model has 3072 quadratic objective terms\n",
      "Model has 67 general constraints\n",
      "Variable types: 3382 continuous, 67 integer (67 binary)\n",
      "Coefficient statistics:\n",
      "  Matrix range     [4e-07, 5e+00]\n",
      "  Objective range  [8e-03, 2e+00]\n",
      "  QObjective range [2e+00, 2e+00]\n",
      "  Bounds range     [4e-04, 2e+01]\n",
      "  RHS range        [4e-03, 5e+00]\n",
      "  GenCon coe range [1e+00, 1e+00]\n",
      "Presolve removed 59 rows and 59 columns\n",
      "Presolve time: 0.73s\n",
      "Presolved: 452 rows, 3390 columns, 330625 nonzeros\n",
      "Presolved model has 3072 quadratic objective terms\n",
      "Variable types: 3256 continuous, 134 integer (134 binary)\n",
      "the layer does not exist\n",
      "the layer does not exist\n",
      "the layer does not exist\n",
      "the layer does not exist\n",
      "the layer does not exist\n",
      "the layer does not exist\n",
      "\n",
      "Root relaxation: objective 4.856702e-10, 437 iterations, 0.21 seconds (0.25 work units)\n",
      "\n",
      "    Nodes    |    Current Node    |     Objective Bounds      |     Work\n",
      " Expl Unexpl |  Obj  Depth IntInf | Incumbent    BestBd   Gap | It/Node Time\n",
      "\n",
      "     0     0    0.00001    0   67          -    0.00001      -     -    4s\n",
      "H    0     0                       0.0000001    0.00001  0.00%     -    8s\n",
      "\n",
      "Explored 1 nodes (437 simplex iterations) in 8.33 seconds (10.53 work units)\n",
      "Thread count was 16 (of 16 available processors)\n",
      "\n",
      "Solution count 1: 5.33192e-10 \n",
      "\n",
      "Optimal solution found (tolerance 1.01e-04)\n",
      "Best objective 5.331912689144e-10, best bound 4.856701707468e-10, gap 0.0000%\n"
     ]
    }
   ],
   "source": [
    "model.setParam(\"OutputFlag\",1)\n",
    "model.setParam(\"TimeLimit\", 100)\n",
    "model.optimize()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 336,
   "id": "fd04668c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "MILP model status: 2, Model solution count: 1, Final solve time: 8.331, Final objval : 0.0000 \n"
     ]
    }
   ],
   "source": [
    "sol_count = f\"{model.solcount:d}\" if hasattr(model, \"solcount\") else \"None\"\n",
    "obj_bound = f\"{model.objbound:.4f}\" if hasattr(model, \"objbound\") else \"failed\"\n",
    "obj_val = f\"{model.objval:.4f}\" if hasattr(model, \"objval\") else \"failed\"     \n",
    "\n",
    "print(f\"MILP model status: {model.Status}, Model solution count: {sol_count}, Final solve time: {model.Runtime:.3f}, Final objval : {obj_val} \")\n",
    "       "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 337,
   "id": "44b932db",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'1'"
      ]
     },
     "execution_count": 337,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sol_count"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 338,
   "id": "f0f2a3f8",
   "metadata": {},
   "outputs": [],
   "source": [
    "if model.solcount > 0:\n",
    "    adv_examples = [model.x[0:num_pixels]]\n",
    "    output_model = [model.x[counter:num_var]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 339,
   "id": "3ccf61a8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "8\n",
      "5\n",
      "[[-1.5621685847816567, -10.869162710137555, -0.5761519891741606, 0.013564872800419767, 0.02629941073018213, -2.344312022620998, -6.491274230228595, -7.035441932900546, -9.478967427172485, -10.037602193069024]]\n"
     ]
    }
   ],
   "source": [
    "print(label)\n",
    "print(adv_label)\n",
    "print(output_model)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2a88ff61",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 340,
   "id": "c30b0a71",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x576 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "adv_examples_img=torch.from_numpy(np.float64(adv_examples)).reshape(1, height, width, channels).permute(0, 3, 1, 2).to('cpu')\n",
    "plot_attack_pred(image, adv_examples_img)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 341,
   "id": "425e942b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor([[[[0., 0., 0.,  ..., 0., 0., 0.],\n",
       "          [0., 0., 0.,  ..., 0., 0., 0.],\n",
       "          [0., 0., 0.,  ..., 0., 0., 0.],\n",
       "          ...,\n",
       "          [0., 0., 0.,  ..., 0., 0., 0.],\n",
       "          [0., 0., 0.,  ..., 0., 0., 0.],\n",
       "          [0., 0., 0.,  ..., 0., 0., 0.]],\n",
       "\n",
       "         [[0., 0., 0.,  ..., 0., 0., 0.],\n",
       "          [0., 0., 0.,  ..., 0., 0., 0.],\n",
       "          [0., 0., 0.,  ..., 0., 0., 0.],\n",
       "          ...,\n",
       "          [0., 0., 0.,  ..., 0., 0., 0.],\n",
       "          [0., 0., 0.,  ..., 0., 0., 0.],\n",
       "          [0., 0., 0.,  ..., 0., 0., 0.]],\n",
       "\n",
       "         [[0., 0., 0.,  ..., 0., 0., 0.],\n",
       "          [0., 0., 0.,  ..., 0., 0., 0.],\n",
       "          [0., 0., 0.,  ..., 0., 0., 0.],\n",
       "          ...,\n",
       "          [0., 0., 0.,  ..., 0., 0., 0.],\n",
       "          [0., 0., 0.,  ..., 0., 0., 0.],\n",
       "          [0., 0., 0.,  ..., 0., 0., 0.]]]], dtype=torch.float64)"
      ]
     },
     "execution_count": 341,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "image- adv_examples_img"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "20fe883f",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "132d7ca8",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
