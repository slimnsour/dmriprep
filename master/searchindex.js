Search.setIndex({docnames:["api","api/dmriprep.config","api/dmriprep.interfaces","api/dmriprep.interfaces.images","api/dmriprep.interfaces.reports","api/dmriprep.interfaces.vectors","api/dmriprep.utils","api/dmriprep.utils.bids","api/dmriprep.utils.images","api/dmriprep.utils.misc","api/dmriprep.utils.vectors","api/dmriprep.workflows","api/dmriprep.workflows.base","api/dmriprep.workflows.dwi","api/dmriprep.workflows.dwi.base","api/dmriprep.workflows.dwi.outputs","api/dmriprep.workflows.dwi.util","changes","index","installation","links","roadmap","usage"],envversion:{"sphinx.domains.c":1,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":1,"sphinx.domains.index":1,"sphinx.domains.javascript":1,"sphinx.domains.math":2,"sphinx.domains.python":1,"sphinx.domains.rst":1,"sphinx.domains.std":1,"sphinx.ext.intersphinx":1,sphinx:56},filenames:["api.rst","api/dmriprep.config.rst","api/dmriprep.interfaces.rst","api/dmriprep.interfaces.images.rst","api/dmriprep.interfaces.reports.rst","api/dmriprep.interfaces.vectors.rst","api/dmriprep.utils.rst","api/dmriprep.utils.bids.rst","api/dmriprep.utils.images.rst","api/dmriprep.utils.misc.rst","api/dmriprep.utils.vectors.rst","api/dmriprep.workflows.rst","api/dmriprep.workflows.base.rst","api/dmriprep.workflows.dwi.rst","api/dmriprep.workflows.dwi.base.rst","api/dmriprep.workflows.dwi.outputs.rst","api/dmriprep.workflows.dwi.util.rst","changes.rst","index.rst","installation.rst","links.rst","roadmap.rst","usage.rst"],objects:{"dmriprep.config":{dumps:[1,1,1,""],environment:[1,2,1,""],execution:[1,2,1,""],from_dict:[1,1,1,""],get:[1,1,1,""],init_spaces:[1,1,1,""],load:[1,1,1,""],loggers:[1,2,1,""],nipype:[1,2,1,""],redirect_warnings:[1,1,1,""],to_filename:[1,1,1,""],workflow:[1,2,1,""]},"dmriprep.config.environment":{cpu_count:[1,3,1,""],exec_docker_version:[1,3,1,""],exec_env:[1,3,1,""],free_mem:[1,3,1,""],nipype_version:[1,3,1,""],overcommit_limit:[1,3,1,""],overcommit_policy:[1,3,1,""],templateflow_version:[1,3,1,""],version:[1,3,1,""]},"dmriprep.config.execution":{bids_description_hash:[1,3,1,""],bids_dir:[1,3,1,""],bids_filters:[1,3,1,""],boilerplate_only:[1,3,1,""],debug:[1,3,1,""],fs_license_file:[1,3,1,""],fs_subjects_dir:[1,3,1,""],init:[1,4,1,""],layout:[1,3,1,""],log_dir:[1,3,1,""],log_level:[1,3,1,""],low_mem:[1,3,1,""],md_only_boilerplate:[1,3,1,""],notrack:[1,3,1,""],output_dir:[1,3,1,""],output_spaces:[1,3,1,""],participant_label:[1,3,1,""],reports_only:[1,3,1,""],run_uuid:[1,3,1,""],templateflow_home:[1,3,1,""],work_dir:[1,3,1,""],write_graph:[1,3,1,""]},"dmriprep.config.loggers":{"default":[1,3,1,""],"interface":[1,3,1,""],cli:[1,3,1,""],init:[1,4,1,""],utils:[1,3,1,""],workflow:[1,3,1,""]},"dmriprep.config.nipype":{crashfile_format:[1,3,1,""],get_linked_libs:[1,3,1,""],get_plugin:[1,4,1,""],init:[1,4,1,""],memory_gb:[1,3,1,""],nprocs:[1,3,1,""],omp_nthreads:[1,3,1,""],parameterize_dirs:[1,3,1,""],plugin:[1,3,1,""],plugin_args:[1,3,1,""],resource_monitor:[1,3,1,""],stop_on_first_crash:[1,3,1,""]},"dmriprep.config.workflow":{anat_only:[1,3,1,""],fmap_bspline:[1,3,1,""],fmap_demean:[1,3,1,""],force_syn:[1,3,1,""],hires:[1,3,1,""],ignore:[1,3,1,""],longitudinal:[1,3,1,""],run_reconall:[1,3,1,""],skull_strip_fixed_seed:[1,3,1,""],skull_strip_template:[1,3,1,""],spaces:[1,3,1,""],use_syn:[1,3,1,""]},"dmriprep.interfaces":{BIDSDataGrabber:[2,2,1,""],BIDSDataGrabberOutputSpec:[2,2,1,""],DerivativesDataSink:[2,2,1,""],images:[3,0,0,"-"],reports:[4,0,0,"-"],vectors:[5,0,0,"-"]},"dmriprep.interfaces.BIDSDataGrabber":{input_spec:[2,3,1,""],output_spec:[2,3,1,""]},"dmriprep.interfaces.DerivativesDataSink":{out_path_base:[2,3,1,""]},"dmriprep.interfaces.images":{ExtractB0:[3,2,1,""],RescaleB0:[3,2,1,""]},"dmriprep.interfaces.images.ExtractB0":{input_spec:[3,3,1,""],output_spec:[3,3,1,""]},"dmriprep.interfaces.images.RescaleB0":{input_spec:[3,3,1,""],output_spec:[3,3,1,""]},"dmriprep.interfaces.reports":{AboutSummary:[4,2,1,""],AboutSummaryInputSpec:[4,2,1,""],SubjectSummary:[4,2,1,""],SubjectSummaryInputSpec:[4,2,1,""],SubjectSummaryOutputSpec:[4,2,1,""],SummaryInterface:[4,2,1,""],SummaryOutputSpec:[4,2,1,""]},"dmriprep.interfaces.reports.AboutSummary":{input_spec:[4,3,1,""]},"dmriprep.interfaces.reports.SubjectSummary":{input_spec:[4,3,1,""],output_spec:[4,3,1,""]},"dmriprep.interfaces.reports.SummaryInterface":{output_spec:[4,3,1,""]},"dmriprep.interfaces.vectors":{CheckGradientTable:[5,2,1,""]},"dmriprep.interfaces.vectors.CheckGradientTable":{input_spec:[5,3,1,""],output_spec:[5,3,1,""]},"dmriprep.utils":{bids:[7,0,0,"-"],images:[8,0,0,"-"],misc:[9,0,0,"-"],vectors:[10,0,0,"-"]},"dmriprep.utils.bids":{collect_data:[7,1,1,""],validate_input_dir:[7,1,1,""],write_derivative_description:[7,1,1,""]},"dmriprep.utils.images":{extract_b0:[8,1,1,""],median:[8,1,1,""],rescale_b0:[8,1,1,""]},"dmriprep.utils.misc":{check_deps:[9,1,1,""]},"dmriprep.utils.vectors":{DiffusionGradientTable:[10,2,1,""],bvecs2ras:[10,1,1,""],calculate_pole:[10,1,1,""],normalize_gradients:[10,1,1,""]},"dmriprep.utils.vectors.DiffusionGradientTable":{affine:[10,4,1,""],b0mask:[10,4,1,""],bvals:[10,4,1,""],bvecs:[10,4,1,""],generate_rasb:[10,4,1,""],generate_vecval:[10,4,1,""],gradients:[10,4,1,""],normalize:[10,4,1,""],normalized:[10,4,1,""],pole:[10,4,1,""],reorient_rasb:[10,4,1,""],to_filename:[10,4,1,""]},"dmriprep.workflows":{base:[12,0,0,"-"],dwi:[13,0,0,"-"]},"dmriprep.workflows.base":{init_dmriprep_wf:[12,1,1,""],init_single_subject_wf:[12,1,1,""]},"dmriprep.workflows.dwi":{base:[14,0,0,"-"],outputs:[15,0,0,"-"],util:[16,0,0,"-"]},"dmriprep.workflows.dwi.base":{init_early_b0ref_wf:[14,1,1,""]},"dmriprep.workflows.dwi.outputs":{init_reportlets_wf:[15,1,1,""]},"dmriprep.workflows.dwi.util":{init_dwi_reference_wf:[16,1,1,""],init_enhance_and_skullstrip_dwi_wf:[16,1,1,""]},dmriprep:{config:[1,0,0,"-"],interfaces:[2,0,0,"-"],utils:[6,0,0,"-"],workflows:[11,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","function","Python function"],"2":["py","class","Python class"],"3":["py","attribute","Python attribute"],"4":["py","method","Python method"]},objtypes:{"0":"py:module","1":"py:function","2":"py:class","3":"py:attribute","4":"py:method"},terms:{"02deb54a823f":1,"121754_aa0b4fa9":1,"182344_950a3a7":1,"1a0":18,"1st":18,"27121_a22e51b47c544980bad594d5e0bb2d04":10,"3dautomask":16,"3dshore":21,"3dunif":16,"44fa":1,"4a11":1,"6b60":1,"6mm":16,"case":[21,22],"class":[0,1,2,3,4,5,10],"default":[1,16,22],"export":1,"final":16,"float":10,"function":[0,1,19,21,22],"import":[1,22],"int":[10,16],"long":18,"new":[1,18,19],"return":10,"static":10,"switch":1,"true":[1,5,7,10],"try":19,ARS:10,For:[19,22],LAS:10,LPS:10,RAS:[10,14],The:[1,10,16,18,19],Then:16,There:19,These:19,Use:[17,22],Using:21,_bidsdatagrabberinputspec:2,_bidsdatagrabberoutputspec:2,_checkgradienttableinputspec:5,_checkgradienttableoutputspec:5,_config:1,_extractb0inputspec:3,_extractb0outputspec:3,_rescaleb0inputspec:3,_rescaleb0outputspec:3,a9272efe69d1:1,abl:22,aboout:21,about:[12,21],aboutsummari:4,aboutsummaryinputspec:4,abov:[14,19,21],absenc:1,access:[1,22],accord:17,acquisit:18,across:[1,8,22],action:1,adapt:[16,18,21],add:[1,17],addit:[19,22],address:18,adher:[17,19],advanc:18,af7d:1,affin:[10,16],afni:[16,19],after:[1,16,19,21],aggreg:22,agnost:18,ahead:19,algorithm:16,alia:[2,3,4,5],all:[1,3,10,12,19,22],allclos:5,alloc:1,allow:[1,19],allowed_ent:2,alpha:21,also:[17,19,21],altern:[1,19],although:22,amazonaw:10,ambiti:19,amd64:19,amplitud:10,analysi:[18,21],analysis_level:[19,22],analyt:[1,18],analyz:21,anat:22,anat_onli:1,anatom:[1,12,17,22],ani:[1,12,18,22],ant:[16,18,19],antsbrainextract:22,api:18,app:[19,22],appear:19,appli:16,applic:[16,18],approach:[19,21],april:18,arg:[1,2],argument:[1,18,19],around:21,arrai:10,artifact:21,aspect:22,assess:22,assum:22,attempt:[17,22],autom:[1,19],automat:18,avail:[1,16,18,19,22],averag:[3,8,14,21],b07ee615a588acf67967d70f5b2402ffbe477b99a316433fa7fdaff4f9dad5c1:1,b0_ix:[3,8,16],b0_threshold:10,b0mask:10,b0s:16,b_scale:10,ba59:1,back:17,bare:19,base:[0,1,2,3,4,5,10,11,13,17,18,19,21],baseinterfaceinputspec:4,bash:19,basi:1,basic:18,batteri:15,becom:18,been:10,befor:[14,18],below:10,best:18,bet:16,better:18,between:21,bia:[16,21],bias_corrected_fil:16,bid:[0,1,2,6,14,18,19,21],bids_description_hash:1,bids_dir:[1,7,22],bids_filt:1,bids_root:22,bids_valid:7,bidsdatagrabb:2,bidsdatagrabberoutputspec:2,bidslayout:1,big:17,binari:[14,19],blob:22,boilerpl:[1,22],boilerplate_onli:[1,22],bound:22,brain:[1,16,18],branch:17,breed:18,bring:17,bspline:22,bugfix:17,build:[1,14,16,17,18,19],build_workflow:1,built:1,bval:[5,10],bvec:[5,10],bvec_norm_epsilon:10,bvecs2ra:10,c3d:19,cach:1,calcul:[10,16,21],calculate_pol:10,call:[1,10,22],can:[1,18,19,21,22],candid:21,capabl:[17,19],categori:1,challeng:18,chang:[1,17],changelog:17,charact:1,chdir:[3,5],check:[5,10,17,19,22],check_dep:9,checkgradientt:5,checkout:19,checkpoint:1,checksum:1,choic:22,chose:19,circleci:17,citat:22,citi:22,classmethod:1,clean:[18,22],cleanenv:22,clear:22,cli:[1,22],client:[1,19],cloud:18,code:[1,12,14,16,17],cohort:22,collect:[1,12],collect_data:7,colon:22,com:[10,19,22],come:19,command:[1,18,19],commerci:18,common:22,commun:21,compar:19,compat:14,complet:22,complex:18,compliant:[1,22],compos:[10,19],comprehend:12,comput:[1,22],concurr:22,config:[0,17,18,22],config_fil:1,configur:[0,18,19],consid:[10,21],consist:1,consumpt:[1,14],contact:19,contain:[1,18,21,22],container_command_and_opt:19,container_imag:19,content:[17,22],continu:[1,17],contrast:16,contribut:17,conveni:1,convers:[1,10],convert:[1,10],coordin:[10,14],copi:1,core:[2,3,4,5],correct:[5,14,16,18,21],correctli:19,correspond:[1,10,22],count:17,countri:22,cover:10,cpu:[1,22],cpu_count:1,crash:22,crashfil:1,crashfile_format:1,crawl:1,creat:[1,12,18,19],crucial:22,current:[1,18,19,21],custom:[2,16,17,22],daemon:19,data:[1,10,16,18,21,22],data_dir:[3,5],dataset:[1,8,16,18,21,22],dataset_descript:1,datasink:[2,15],deal:3,debian:19,debug:[1,22],decai:3,decemb:18,defin:22,definit:22,delimit:22,demean:22,denois:21,depend:[9,18,22],deploi:[17,19],deploy:17,deriv:[1,12,15,18,21],deriv_dir:7,derivatives_path:19,derivativesdatasink:2,describ:22,descript:[1,19],design:[1,18,22],detail:22,develop:[18,22],dict:1,dictionari:1,diffus:[10,12,18,21],diffusiongradientt:10,dilat:16,dimens:8,dipi:21,dir:22,direct:12,directori:[1,22],disabl:22,disk:[1,22],displac:21,distort:[1,14,18,21],distribut:16,dmri:[14,18,21,22],dmriprep:[0,17,19,22],dmriprep_named_opt:19,doc:[17,19],docker:[1,17,18,22],dockerfil:[17,19],dockerout:22,document:[17,18,19],doe:[1,22],doing:21,don:22,done:22,drift:21,drop:19,dump:1,dwi:[0,3,5,8,10,11,12,17,22],dwi_fil:[5,10,14,16],dwi_mask:[3,14,16],dwi_refer:14,dwi_reference_wf:16,each:[1,12,16,18,22],earli:[1,14,21],earlier:21,early_b0ref_wf:14,easi:18,easili:[1,18],eddi:[18,21],edu:22,effect:22,either:1,element:16,enabl:[1,22],encompas:18,engin:[1,19],enh:17,enhanc:16,enhance_and_skullstrip_dwi_wf:16,enlist:1,ensur:[5,18,22],entiti:[12,22],environ:[1,18,22],equip:18,error:[1,22],estim:[1,21],etc:[1,18],evalu:21,even:22,event:22,everi:1,exact:22,exampl:[1,3,5,10,19,22],exclus:21,exec_docker_vers:1,exec_env:[1,7],execut:[1,12,17,18,19],exercis:17,exist:[1,22],exit:22,expect:21,experiment:22,explor:21,extern:18,extra:19,extract:[1,3,8,16,21],extract_b0:[3,8],extractb0:3,eye:10,fact:22,fals:[1,10],field:[1,16,21,22],fieldmap:[1,18],file:[1,10,14,16,17,19,22],filenam:[1,10],filesystem:1,filetyp:10,filter:[1,18],first:[1,17,19,22],fit:[18,22],fix:[1,17,22],flag:[1,22],flake8:17,flat:1,fmap:22,fmap_bsplin:1,fmap_demean:1,fmriprep:17,folder:[1,12,22],follow:[19,21,22],forc:22,force_syn:1,forkserv:1,form:22,format:[1,18],found:[1,19,22],frame:10,framewis:21,framework:19,free:[1,19,22],free_mem:1,freesurf:[1,12,18,19],freesurfer_hom:22,from:[1,3,8,16,18,19,21,22],from_dict:1,from_fil:[3,4,5],fs_licens:22,fs_license_fil:1,fs_subjects_dir:1,fsl:[16,19],full:[1,10,12,21],full_spher:5,fullds005:22,fund:22,further:[21,22],futur:[21,22],g4e9d623:[1,22],gener:[1,4,16,18,19,22],generate_rasb:10,generate_vecv:10,get:[1,10,19,21,22],get_linked_lib:1,get_plugin:1,gibb:21,github:[17,22],given:[1,10],googl:[1,18],gradient:[5,10,14,21],gradients_rasb:14,graph:[1,12,14,16,22],grid:22,guid:21,guidelin:17,habitu:19,had:16,half:10,hand:19,handl:[5,7,8,18,19],happen:1,hard:1,harvard:22,has:[1,18,19],hash:1,have:[10,19,21,22],head:[18,21],hello:19,help:[18,22],hemispher:10,heurist:1,high:18,highli:22,hire:[1,22],histogram:16,hmc:21,hoc:18,holder:21,home:[1,22],host:[19,22],how:19,hpc:18,html:[1,10,16,22],http:[10,19,22],hub:19,idea:19,identif:21,identifi:[1,19,21,22],idiosyncrasi:18,ignor:[1,22],imag:[0,2,6,10,16,19,21,22],implement:[1,18,21],impli:22,improv:[19,22],imput:21,in_bval:[5,14],in_bvec:[5,14],in_fil:[3,8,16],in_rasb:5,includ:[21,22],increas:22,index:16,indic:[16,22],individu:16,infer:18,info:1,inform:[0,1,12,22],infrastructur:[17,19],inhomogen:21,init:1,init_dmriprep_wf:12,init_dwi_reference_wf:[14,16],init_early_b0ref_wf:14,init_enhance_and_skullstrip_dwi_wf:16,init_enhance_and_skullstrip_wf:16,init_reportlets_wf:[14,15],init_single_subject_wf:12,init_spac:1,initi:[1,16,17],input:[2,3,4,5,7,8,12,14,16,22],input_bids_path:19,input_spec:[2,3,4,5],instal:[1,18,22],instanc:[1,12],instead:17,instruct:19,integr:[17,21],intend:22,intens:[8,16,21],interfac:[1,18,19],intermedi:[1,22],intern:14,interpret:[18,19],intersect:16,intervent:18,inu:[16,21],inventori:18,involv:18,issu:1,iter:[1,16],its:16,januari:18,join:1,json:[1,22],just:10,keep:[1,22],kei:22,kernel:[1,19],keyword:22,known:19,kwarg:[2,4],label:[12,22],laptop:18,larg:18,last:[8,10],latest:[16,22],latex:[1,22],layout:[1,22],lead:21,least:22,left:1,length:10,less:1,level:[1,19,22],leverag:21,librari:[1,18],licens:[1,18,19],lie:10,lightweight:19,like:[1,21],limit:[1,19,22],line:[1,18,19],linear:21,lineno:1,link:1,lint:17,linux:[1,19],list:[1,10,12,16,19,22],load:[1,17],loadtxt:5,local:[17,22],locat:[10,15,22],log:[0,18,22],log_dir:1,log_level:1,logger:1,logitudin:1,longitudin:[1,22],look:1,loos:16,low:[10,22],low_mem:1,machin:19,magic:1,mai:[12,16,18,22],maint:17,maintain:1,mainten:17,major:21,make:[9,10,19,21],makefil:17,manag:1,mani:19,manual:[18,22],map:18,march:18,mark:17,markdown:1,mask:[10,14,16,22],mask_fil:[3,8,16],master:22,mathemat:16,maximum:[16,22],maxtasksperchild:1,md_only_boilerpl:1,mean:[1,22],measur:21,median:[8,16,22],mem:22,mem_gb:16,mem_mb:22,memori:[1,22],memory_gb:[1,22],merg:21,messag:[1,19],metal:19,method:[0,19,22],methodolog:16,mgh:22,mgr:1,millimet:22,minim:[1,17],misc:[0,6],miscellan:9,mode:1,model:[18,21],modifi:[19,22],modul:[0,1,2,6,11,13],monitor:[1,22],more:19,morpholog:16,most:[10,19],motion:[18,21],mri:[12,18],multi:19,multiproc:1,multiprocess:1,multithread:1,must:[1,10,19,22],n4biasfieldcorrect:16,n_cpu:22,name:[14,15,16,18,19],named_opt:19,ndarrai:10,necessari:19,need:[1,12],neurodock:19,neuroimag:[18,19],neuroscientist:18,neurostar:18,newer:18,newrasb:5,next:19,nifti:[1,14,16],nii:[3,5],niprep:[19,22],nipyp:[1,2,3,4,5,18,19,22],nipype_vers:1,niworkflow:[2,7,16],nmr:22,node:[1,16,22],non:[1,10,21,22],none:[1,2,3,4,5,8,10],nonstandard:1,norm:10,norm_val:10,norm_vec:10,normal:[10,14,22],normalize_gradi:10,note:22,notrack:[1,22],notrecommend:22,novemb:18,now:22,nproc:[1,22],nthread:22,number:[1,10,12,16,22],numer:18,numpi:10,oasis30:1,object:[1,10],obtain:[19,22],occurr:22,oesteban:1,oldrasb:5,omp:22,omp_nthread:[1,16,22],onc:19,one:[12,16,22],ones:10,onli:[1,22],onlin:22,oper:[1,10,19,22],oppos:12,opt:[1,22],option:[1,17,18,19],orchestr:14,org:[18,22],organ:12,orient:22,origin:22,other:[0,16,18,19,21],otherwis:1,othewis:10,our:19,out:[1,17,19,22],out_path:8,out_path_bas:2,out_rasb:5,out_report:16,outcom:22,outlier:21,output:[0,1,5,11,13,14,16,18,19,22],output_dir:[1,22],output_spac:[1,22],output_spec:[2,3,4,5],over:[1,3],overcommit:1,overcommit_limit:1,overcommit_polici:1,packag:[0,18,19],page:17,pair:10,pandoc:22,parallel:[1,16],paramet:[1,10,12,16,22],parameter:1,parameterize_dir:1,part:22,particip:[1,22],participant_id:1,participant_label:[1,7,22],particular:[1,18],pass:[1,22],patch:2,path:[1,10,14,19,22],pathlik:12,pca:21,pdf:[12,14,16],per:22,perform:[12,16,18],pickl:1,pip:19,pipelin:[12,18,21],plan:[1,18],platform:1,pleas:[19,22],plugin:[1,17,22],plugin_arg:1,png:[12,14,16],point:[10,21,22],pole:[5,10],polici:1,popul:12,popular:19,popylar:[17,22],port:21,posit:18,posix:1,posixpath:1,possibl:[1,21,22],pre_mask:16,preambl:19,prefix:22,prepar:[12,18,22],preprocess:[1,12,14,18],present:9,previou:22,price:19,princip:22,principl:17,probabl:19,process:[1,12,16,17,18,21,22],produc:19,program:[18,22],project:1,properti:10,provid:[18,22],pub:10,pull:19,pybid:22,pypi:19,python:[1,18],qsiprep:21,qualiti:[18,22],queri:18,raise_error:10,raise_inconsist:10,raise_insuffici:1,ram:1,random:22,rapid:18,rasb:[10,14],rasb_fil:10,raw:[10,21],raw_ref_imag:16,read:[1,19],readi:19,real:22,reason:[19,22],recent:10,recommend:[18,22],recon:[1,12,22],reconal:22,reconstruct:[1,22],record:22,redirect:1,redirect_warn:1,reduc:22,ref_imag:16,ref_image_brain:16,refactor:17,refer:[1,10,14,16,17,22],regard:1,regardless:12,regist:[16,22],registr:[18,21],registri:17,regular:1,rel:19,releas:[17,21],remov:[1,16,17,22],reorient:10,reorient_rasb:10,replac:[1,7],replic:22,report:[0,1,2,12,15,17,22],reportlet:[1,4,15,16,22],reportlets_dir:15,reportlets_wf:15,reports_onli:1,repres:1,represent:[1,17],reproduc:18,requir:[19,22],rerun:22,res:[3,22],resampl:22,rescal:[3,8,16,21],rescale_b0:[3,8],rescaleb0:3,research:18,resolut:22,resourc:[1,22],resource_monitor:[1,3,4,5],respons:[0,18],result:[1,10,18,21,22],resultinb:10,retval:1,reus:22,revis:17,rician:21,right:15,ring:21,road:18,robust:18,roll:17,root:[1,22],rootlogg:1,rstudio:10,rtol:5,run:[1,3,5,12,16,17,19,21,22],run_reconal:1,run_unique_id:1,run_uuid:[1,22],runtim:[1,22],same:19,sampl:22,save:17,sbref:22,scale:14,scan:17,script:17,sdc:22,sdcflow:21,search:22,section:[0,18,19],secur:19,see:[1,19,22],seed:[1,22],select:[1,22],send:22,sent:19,sentri:17,separ:[12,22],septemb:18,seri:[3,22],serv:[17,21],session:[1,12],set:[1,12,15,17,18,19,22],sever:[12,21],sfm:21,sha256:1,shape:10,share:[1,19],sharpen:16,shell:[10,21],shorelin:21,should:[1,19,21,22],show:[17,19,22],signal:[3,8,16,21],simpleinterfac:[2,3,4,5],simpli:22,singl:[1,12,16,21,22],singleton:1,singular:[18,22],skip:22,skiprow:5,skull:[1,16,22],skull_strip_fixed_se:1,skull_strip_templ:[1,22],skull_stripped_fil:16,skullstrip:17,slicetim:22,sloppi:[1,22],small:17,smoke:17,smriprep:17,snowbal:18,softwar:[18,19,21,22],some:[1,17,19,22],someth:19,sourc:[12,14,16],space:[1,22],spatial:[1,22],spatialrefer:1,spec:4,specif:[0,17,18],specifi:[19,22],speed:22,spend:21,sphere:[10,16],spline:[1,22],sprint:17,squar:[10,22],stage:22,stake:21,standard:[1,16,18,22],start:[1,17,19],state:18,statist:22,step:[1,16,17,18,19,21],stop:[1,22],stop_on_first_crash:1,store:[1,15,22],str:[1,3,5,12,16],stream:19,string:1,strip:[1,16,22],structur:[1,10,16,19,22],sub:[1,12,22],subject:[1,12,22],subject_id:12,subjects_dir:12,subjectsummari:4,subjectsummaryinputspec:4,subjectsummaryoutputspec:4,submit:18,submm:22,submodul:[0,18],suboptim:1,subpackag:[0,18],subprocess:1,successfulli:22,suffix:22,summaryinterfac:4,summaryoutputspec:4,support:18,sure:[9,19],surfac:[1,18],surfer:22,suscept:[1,14,18,21],svg:[12,14,16],syn:18,system:[9,19,22],t1w:[21,22],tabl:[5,10,14],tacc:19,tag:17,take:[16,22],target:[1,17,18],task:1,task_id:17,technolog:18,templat:[1,12,22],templateflow:1,templateflow_hom:1,templateflow_vers:1,temporari:22,tenant:19,term:18,termin:19,test:[17,21,22],text:1,thei:21,them:[16,17,21],therefor:19,thesee:21,thi:[1,9,12,16,17,18,19,21,22],thp0005:1,thp002:1,thread:[16,22],threshold:10,through:22,time:[1,3,21,22],tip:19,tmp:1,tmpdir:[3,5],to_filenam:[1,10],tolist:10,toml:1,took:19,tool:[1,3,18,19,22],top:22,traceback:10,track:[1,17,18],traitedspec:4,transform:10,transpar:18,travisci:17,treat:22,trick:[1,19],tsv:5,two:[12,19],txt:[1,22],type:10,typo:17,ubuntu:19,uncompress:1,under:[1,19],uniform:21,uniqu:1,unit:[10,17],unless:22,updat:[17,18],upon:18,upper:22,usag:[0,18,19],use:[16,18,19,21,22],use_plugin:22,use_syn:1,used:[1,19,21,22],useful:1,user:[1,19,21],uses:22,using:[1,8,12,16,19,21,22],usr:1,util:[0,1,11,13,17,18],uuid:22,valid:[16,18,19,21,22],validate_input_dir:7,validation_report:16,valu:[1,10,14,16,22],valueerror:10,variabl:[1,22],varieti:18,variou:21,vec:10,vector:[0,2,6,14,17],verbos:[1,22],veri:22,version:[1,7,16,17,18,19,22],via:1,virtual:[1,18,19],visit:19,visual:22,volum:[3,8,10,16,21],vstack:10,vvv:22,wai:[19,21],want:22,warn:1,well:21,were:19,what:18,when:[1,22],whenev:1,where:[1,10,21,22],whether:[1,10,16,22],which:[1,10,18,19,22],whole:18,wide:1,within:[19,22],without:[18,22],work:[1,17,19,22],work_dir:[1,22],workdir:22,workflow:[0,1,9,17,18,19],world:[19,22],would:21,write:[1,10,15,22],write_derivative_descript:7,write_graph:1,written:[1,19],www:22,xxxxx:22,you:[19,22],your:[19,22],zenodo:17,zero:10},titles:["Library API (application program interface)","dmriprep.config package","dmriprep.interfaces package","dmriprep.interfaces.images module","dmriprep.interfaces.reports module","dmriprep.interfaces.vectors module","dmriprep.utils package","dmriprep.utils.bids module","dmriprep.utils.images module","dmriprep.utils.misc module","dmriprep.utils.vectors module","dmriprep.workflows package","dmriprep.workflows.base module","dmriprep.workflows.dwi package","dmriprep.workflows.dwi.base module","dmriprep.workflows.dwi.outputs module","dmriprep.workflows.dwi.util module","What\u2019s new?","dMRIPrep","Installation","&lt;no title&gt;","Development road map","Usage"],titleterms:{"1a0":17,"1st":21,"long":21,"new":17,The:22,analyt:22,ant:22,api:0,applic:0,april:21,argument:22,base:[12,14],bid:[7,22],cloud:19,command:22,commerci:19,config:1,configur:[1,22],contain:19,content:18,correct:22,decemb:17,depend:19,develop:21,distort:22,dmriprep:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18],docker:19,dwi:[13,14,15,16],environ:19,execut:22,extern:19,fieldmap:22,filter:22,format:22,freesurf:22,googl:22,handl:22,hpc:19,imag:[3,8],instal:19,interfac:[0,2,3,4,5],januari:17,laptop:19,librari:0,licens:22,line:22,log:1,mai:21,manual:19,map:21,march:21,misc:9,modul:[3,4,5,7,8,9,10,12,14,15,16],name:22,novemb:17,option:22,other:[1,22],output:15,packag:[1,2,6,11,13],perform:22,plan:21,posit:22,prepar:19,preprocess:22,program:0,python:19,queri:22,recommend:19,registr:22,report:4,respons:1,road:21,section:1,septemb:[17,21],singular:19,specif:22,submodul:[2,6,11,13],subpackag:11,surfac:22,syn:22,target:21,technolog:19,term:21,track:22,usag:[1,22],util:[6,7,8,9,10,16],vector:[5,10],version:21,what:17,workflow:[11,12,13,14,15,16,22]}})