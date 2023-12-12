#----- Obtains the category descriptions -----#

import numpy as np

# Returns the level 1 label descriptions
def get_level_1_descriptions():
    descriptions = {
    1: "The production and management of crops and meat, the breeding of agricultural animals and plants, the development of resources like soil and forests, and the processing of food.",
    4: "The theory and practice of the planning, managing, marketing and appraisal of small and large business enterprises. The systematic study of production, conservation and allocation of financial and human resources, as well as the preparation of individuals for administration and management in the hospitality and tourism industry.",
    7: "The science and practice of educating the population.", 
    8: "The application of mathematical and scientific principles to the solution of practical problems.",
    9: "Prepares individuals to practice as professionals, technicians/technologists and assistants in the health care professions and also focuses on the study of related clinical sciences.",
    11: "The structures and use in speech, writing and various other aspects of the literatures of the official languages of South Africa, as well as areas of study relating to other important languages, inter alia other African languages, Asian languages, European languages, Classical languages and South African sign language.",
    13: "The biological sciences and the non-clinical biomedical sciences.", 
    14: "The scientific inanimate objects, processes of matter and energy and associated phenomena.", 
    17: "The critical analysis of and reflection on the ideas, theories, categories, concepts and methods for describing and evaluating human experience and reality. Religion and different theologies, namely Buddhism, Christianity, Hinduism, Islam, Judaism and African indigenous religions.",
    18: "The behaviour of individuals, independently or collectively, and the physical and environmental bases of mental, emotional and neurological development and processes.",
    20: "Social sciences, social institutions, and social behaviour, as well as the interpretation of past events, issues and cultures."
    }
    return np.array(list(descriptions.values()))

# Returns the level 2 label descriptions
def get_level_2_descriptions():
    descriptions = {
    101: "Manage agricultural businesses \
        and agriculturally related operations within diversified corporations. Includes\
        instruction in agriculture, agricultural specialisation, business management,\
        accounting, finance, marketing, planning, human resources management, and\
        other managerial responsibilities. application of economics to the analysis of\
        resource allocation, productivity, investment, and trends in the agricultural sector,\
        both domestically and internationally. economics and\
        related subfields as well as applicable agricultural fields. \
        Manage farms, reserves and\
        similar enterprises. applicable agricultural specialisation,\
        business management, accounting, taxation, capitalisation, purchasing,\
        government programmes and regulations, operational planning and budgeting,\
        contracts and negotiation, and estate planning. sell agricultural products and\
        supplies, provide support services to agricultural enterprises, and purchase and\
        market agricultural products. \
        Basic business management,\
        marketing, retailing and wholesaling operations, and applicable principles of\
        agriculture and agricultural operations. \
        Perform specialised support\
        functions related to agricultural business offices and operations, and to operate\
        agricultural office equipment, software and information systems. Includes\
        instruction in basic agricultural business principles, office management,\
        equipment operation, standard software, and database management. \
        Provide referral, consulting, technical\
        assistance, and educational services to gardeners, farmers, ranchers,\
        agribusinesses, and other organisations. basic agricultural\
        sciences, agricultural business operations, pest control, adult education methods,\
        public relations, applicable state laws and regulations, and communication skills.", 
    
    108:'The\
        breeding, cultivation and production of agricultural plants, and the production,\
        processing and distribution of agricultural plant products. Includes instruction in\
        the plant sciences, crop cultivation and production, and agricultural and food\
        products processing.\
        The chemical, physical and biological\
        relationships of crops and the soils nurturing them. Includes instruction in the\
        growth and behaviour of agricultural crops, the development of new plant\
        varieties, and the scientific management of soils and nutrients for maximum plant\
        nutrition, health and productivity.\
        Cultivation\
        of garden and ornamental plants, including fruits, vegetables, flowers, and\
        landscape and nursery crops. Includes instruction in specific types of plants, such\
        as citrus; breeding horticultural varieties; physiology of horticultural species; and\
        the scientific management of horticultural plant development and production\
        through the life cycle.\
        Genetics and genetic\
        engineering to the improvement of agricultural plant health, the development of\
        new plant varieties, and the selective improvement of agricultural plant\
        populations. Includes instruction in genetics, genetic engineering, population\
        genetics, agronomy, plant protection, and biotechnology.\
        The\
        control of animal and weed infestation of domesticated plant populations,\
        including agricultural crops; the prevention/reduction of attendant economic loss;\
        and the control of environmental pollution and degradation related to pest\
        infestation and pest control measures. Includes instruction in entomology,\
        applicable animal sciences, plant pathology and physiology, weed science, crop\
        science, and environmental toxicology.\
        Rangelands, arid regions,\
        grasslands, and other areas of low productivity, as well as the principles of\
        managing such resources for maximum benefit and environmental balance.\
        Includes instruction in livestock management, wildlife biology, plant sciences,\
        ecology, soil science and hydrology.\
        The \
        breeding, cultivation and production of grapevines for wine and table grapes, as\
        well as molecular biology, genetics, metabolomics and genetic engineering for the\
        improvement of wine and table grape cultivars and rootstocks. Includes\
        instruction in cultivar science; grapevine physiology; molecular aspects of key\
        processes; the development of new cultivars and clones; terroir; biotechnology;\
        and in vitro tissue culture technologies and the scientific management of\
        grapevine plant development and vineyard cultivation practices.',

    401:'Plan, organise, direct and \
        control the functions and processes of a firm or organisation. Includes instruction\
        in management theory, human resources management and behaviour, accounting\
        and other quantitative methods, purchasing and logistics, organisation and\
        production, marketing and business decision-making.\
        manage and/or administer the\
        processes by which a firm or organisation contracts for goods and services to\
        support its operations, as well as contracts it to sell to other firms or organisations.\
        Includes instruction in contract law, negotiations, buying procedures, government\
        contracting, cost and price analysis, vendor relations, contract administration,\
        auditing and inspection, relations with other firm departments, and applications to\
        special areas such as high-technology systems, international purchasing, and\
        construction.\
        manage and coordinate all\
        logistical functions in an enterprise, ranging from acquisitions to receiving and\
        handling, through internal allocation of resources to operations units, to the\
        handling and delivery of output. Includes instruction in acquisitions and\
        purchasing, inventory control, storage and handling, just-in-time manufacturing,\
        logistics planning, shipping and delivery management, transportation, quality\
        control, resource estimation and allocation, and budgeting.\
        supervise and manage the\
        operations and personnel of business offices and management-level divisions.\
        Includes instruction in employee supervision; management and labour relations;\
        budgeting; scheduling and coordination; office systems operation and\
        maintenance; office records management; organisation, and security; office\
        facilities design and space management; preparation and evaluation of business\
        management data; and public relations.\
        manage and direct the physical\
        and/or technical functions of a firm or organisation, particularly those relating to\
        development, production and manufacturing. Includes instruction in principles of\
        general management, corporate governance, manufacturing and production\
        systems, plant management, equipment maintenance management, production\
        control, industrial labour relations and skilled trades supervision, strategic\
        manufacturing policy, systems analysis, productivity analysis and cost control, and\
        materials planning.\
        manage the business affairs of\
        non-profit corporations, including foundations, educational institutions,\
        associations and other such organisations, and public agencies and governmental\
        operations. Includes instruction in business management, principles of public\
        administration, principles of accounting and financial management, human\
        resources management, taxation of non-profit organisations, and business law as\
        applied to non-profit organisations.\
        supervise and monitor customer\
        service performance and manage frontline customer support services, call\
        centres/help desks, and customer relations. Includes instruction in customer\
        behaviour; developing and using customer service databases; user surveys and\
        other feedback mechanisms; strategic and performance planning and analysis;\
        operations management; personnel supervision; and communications and\
        marketing skills.\
        plan, manage, supervise, and\
        market electronic business operations, products, and services provided online via\
        the Internet. Includes instruction in business administration, information\
        technology, information resources management, web design, computer and\
        Internet law and policy, computer privacy and security, e-trading, insurance,\
        electronic marketing, investment capital planning, enterprise operations,\
        personnel supervision, contracting, and product and service networking.\
        plan, administer and coordinate\
        physical transportation operations, networks and systems. Includes instruction in\
        transportation systems and technologies; traffic logistics and engineering; multiand intermodal transportation systems; regional integration; facilities design and\
        construction; transportation planning and finance; demand analysis and\
        forecasting; carrier management; behavioural issues; transportation policy and\
        law; intelligent systems; applications to aviation, maritime, rail and highway\
        facilities and systems.\
        organise and manage resources\
        (e.g. people) in such a way that the project is completed within defined scope,\
        quality, time and cost constraints.',

    404:'The production,\
        conservation and allocation of resources in conditions of scarcity, together with\
        the organisational frameworks related to these processes. Includes instruction in\
        economic theory, micro- and macroeconomics, comparative economic systems,\
        money and banking systems, international economics, quantitative analytical\
        methods, and applications to specific industries and public policy issues.\
        application of economic principles and\
        Analytical techniques to the study of particular industries, activities, or the\
        exploitation of particular resources. Includes instruction in economic theory;\
        microeconomic analysis and modelling of specific industries, commodities; the\
        economic consequences of resource allocation decisions; regulatory and\
        consumer factors; and the technical aspects of specific subjects as they relate to\
        economic analysis.\
        Application of economics principles to the\
        analysis of the organisation and operation of business enterprises. Includes\
        instruction in monetary theory, banking and financial systems, theory of\
        competition, pricing theory, wage and salary/incentive theory, analysis of markets,\
        and applications of econometrics and quantitative methods to the study of\
        particular business and business problems.\
        Mathematical and\
        statistical analysis of economic phenomena and problems. Includes instruction in\
        economic statistics, optimisation theory, cost/benefit analysis, price theory,\
        economic modelling, and economic forecasting and evaluation.\
        The economic\
        development process and its application to the problems of specific countries and\
        regions. Includes instruction in economic development theory, industrialisation,\
        land reform, infrastructural development, investment policy, the role of\
        governments and business in development, international development\
        organisations, and the study of social, health, and environmental influences on\
        economic development.\
        Analysis of\
        international commercial behaviour and trade policy. Includes instruction in\
        international trade theory, tariffs and quotas, commercial policy, trade factor flows,\
        international finance and investment, currency regulation and trade exchange\
        rates and markets, international trade negotiation, and international payments and\
        accounting policy.\
        Application of economic concepts and\
        methods to the analysis of issues such as air and water pollution, land use\
        planning, waste disposal, invasive species and pest control, conservation policies,\
        and related environmental problems. Includes instruction in cost-benefit analysis;\
        environmental impact assessment; evaluation and assessment of alternative\
        resource management strategies; policy evaluation and monitoring; and\
        descriptive and analytic tools for studying how environmental developments affect\
        the economic system.',

    702: 'The curriculum and related instructional\
        processes and tools, and that may prepare individuals to serve as professional\
        curriculum specialists. Includes instruction in curriculum theory, curriculum design\
        and planning, instructional material design and evaluation, curriculum evaluation,\
        and applications to specific subject matter, programmes or educational levels.',

    703: 'General principles and techniques of\
        administering a wide variety of schools and other educational organisations and\
        facilities, supervising educational personnel at the school or staff level, and that\
        may prepare individuals as general administrators and supervisors.\
        Plan, supervise, and manage\
        programmes for exceptional students and their parents. Includes instruction in\
        special education theory and practice, special education programme\
        development, evaluation and assessment in special education, relevant law and\
        regulations, managing individual education plans, problems of low- and highdisability students, mainstreaming, special education curricula, staff management,\
        parent education, communications and community relations, budgeting, and\
        professional standards and ethics.\
        Principles and techniques of administering\
        programmes and facilities designed to serve the basic education needs of\
        undereducated adults, or the continuing education needs of adults seeking further\
        or specialised instruction, and that prepares individuals to serve as administrators\
        of such programmes. Includes instruction in adult education principles,\
        programme and facilities planning, personnel management, community and client\
        relations, budgeting and administration, professional standards, and applicable\
        laws and policies.\
        Supervise instructional and support\
        personnel at the school building, facility or staff level. Includes instruction in the\
        principles of staffing and organisation, the supervision of learning activities,\
        personnel relations, administrative duties related to departmental or unit\
        management, and specific applications to various educational settings and\
        curricula.\
        Principles and practice of administration in\
        higher education institutions, the study of higher education as an object of applied\
        research, and which may prepare individuals to function as administrators in such\
        settings. Includes instruction in higher education economics and finance; policy\
        and planning studies; curriculum; faculty and labour relations; higher education\
        law; college student services; research on higher education; institutional research;\
        marketing and promotion; and issues of evaluation, accountability and philosophy.\
        educational administration at the pre-primary\
        and primary school stages, and prepares individuals to serve as principals in\
        schools which focus on these levels. Includes instruction in programme and\
        facilities planning, budgeting and administration, public relations, human\
        resources management, childhood and pre-adolescent growth and development,\
        counselling skills, applicable law and regulations, school safety and discipline,\
        policy studies, cultural context and professional standards and ethics.\
        educational administration at the secondary\
        school (grades 8-12) levels, and prepares individuals to serve as principals of\
        these schools. Includes instruction in secondary school education, programme\
        and facilities planning, budgeting and administration, public relations, human\
        resources management, adolescent growth and development, counselling skills,\
        applicable law and regulations, school safety and discipline, policy studies,\
        cultural context and professional standards and ethics.\
        leading and managing multi-school educational\
        Systems and school districts, and prepares individuals to serve as systems\
        administrators and district superintendents. Includes instruction in educational\
        administration; education of students at various levels; system planning and\
        budgeting; educational policy; educational law and regulations; public relations;\
        professional standards and ethics; and applications to specific issues, cultural\
        context, and geographic locales.', 

    806: 'Systems\
        employing chemical processes, such as chemical reactors, kinetic systems,\
        electrochemical systems, energy conservation processes, heat and mass transfer\
        systems, and separation processes; and the applied analysis of chemical\
        problems such as corrosion, particle abrasion, energy loss, pollution and fluid\
        mechanics.',

    809: 'Electrical,\
        electronic and related communications systems and their components, including\
        electrical power generation systems; and the analysis of problems such as\
        superconductor, wave propagation, energy storage and retrieval, and reception\
        and amplification.',

    815: 'Physical\
        systems used in manufacturing and end-product systems used for specific uses,\
        including machine tools, jigs and other manufacturing equipment; stationary\
        power units and appliances; engines; self-propelled vehicles; housings and\
        containers; hydraulic and electric systems for controlling movement; and the\
        integration of computers and remote control with operating systems.\
        Physical\
        systems that rely on the integration of mechanisms, sensors, actuators, controllers\
        and software, including process and machine control, pneumatic, hydraulic and\
        electric systems for controlling movement; and the integration of computers and\
        remote control with operating systems.',

    907: 'The health care of operating crews\
        and passengers of air and space vehicles, including support personnel. Includes\
        instruction in special conditions of physical and psychological stress, emergency\
        medical procedures, adaptive systems and artificial environments.\
        the delivery of skilled medical care\
        to patients suffering from allergic, asthmatic and immunologic diseases.\
        the application of anaesthesia for\
        General and specialised surgery and obstetrics, critical patient care and the care\
        of pain problems. Includes instruction in surgical procedures, current monitoring\
        procedures, fluid therapy, pain management, diagnostic and therapeutic\
        procedures outside the operating room and operating room safety.\
        the natural history of cardiovascular\
        disorders in adults and the diagnosis and treatment of diseases of the heart and\
        blood vessels. Includes instruction in coronary care, diagnostic testing and\
        evaluation, invasive and non-invasive therapies and pacemaker follow-up.\
        the diagnosis and management of\
        diseases and disorders of the cardiovascular and cardiopulmonary systems of\
        infants, children and adolescents. Includes instruction in related public health and\
        community medicine issues.\
        The use and interpretation of\
        chemical analyses, chemistry diagnoses and management, and the supervision of\
        chemical pathology labs. Includes instruction in basic and applied analytical\
        chemistry pathology, instrumentation data processing and chemical laboratory\
        management.\
        the diagnosis and non-surgical\
        treatment of diseases and abnormalities affecting the nervous system and nerve\
        tissue in adults.\
        Study, prevent and manage\
        neurological disorders of children including disorders requiring surgical, medical\
        and psychiatric management. Includes instruction in neurophysiology,\
        neuropathology, neuroradiology, neuro-ophthalmology, psychiatry, rehabilitation,\
        neurological surgery, paediatric neurodevelopment, and psycho-social support\
        and counselling.\
        The diagnosis, treatment, and\
        prevention of mental, emotional, behavioural and neurological disorders. Includes\
        instruction in psychotherapy, neuropsychiatry, liaison mental health, addictions\
        mental health, family counselling, referral, clinical diagnosis, and practice\
        management.\
        The diagnosis and treatment of\
        mental, emotional and behavioural disorders of infancy, early childhood and\
        adolescence.\
        The surgical care of patients with\
        anorectal and colonic diseases. Also includes instruction in diagnostic and\
        therapeutic colonoscopy.\
        The administration of anaesthesia\
        to patients with acute, chronic or long-term illness and who have multiple organ\
        system derangements. Includes instruction in high-risk and trauma procedures,\
        respiratory therapy and biomedical engineering.\
        The management of care for\
        patients with acutely life-threatening conditions which may include multiple organ\
        failure. Includes instruction in the management of critical care units, emergency\
        procedures and post-discharge care of former critical care patients.\
        surgical procedures for patients\
        with multiple trauma, critical illness, patients on life support and elderly or very\
        young patients with disease complications.\
        the delivery of specialised care to\
        patients with diseases of the skin, hair, nails and mucous membranes. In\
        instruction in dermatologic surgical procedures, histopathology of skin disease,\
        cutaneous allergies, sexually transmitted diseases, and diagnostic and\
        therapeutic techniques.\
        The clinical and microscopic\
        diagnosis and analysis of skin diseases and disorders. Includes instruction in\
        laboratory administration and the supervision and training of support personnel.',
    
    908: 'Knowledge, techniques and procedures for promoting health, providing care for\
        sick, disabled, informed, or other individuals or groups. Includes instruction in the\
        administration of medication and treatments, assisting a physician during\
        treatments and examinations, referring patients to physicians and other health\
        specialists, and planning education for health maintenance.\
        manage nursing personnel\
        and services in hospitals and other health care delivery agencies. Includes\
        instruction in principles of health care administration, resource and financial\
        management, health care law and policy, medical personnel management, and\
        managed care operations.\
        provide general care for\
        adult patients. Includes instruction in adult primary care, adult pathophysiology,\
        clinical management of medication and treatments, patient assessment and\
        education, patient referral, and planning adult health maintenance programmes.\
        administer anaesthetics and\
        provide care for patients before, during and after anaesthesia. Includes\
        instruction in the biochemistry and physics of anaesthesia, advanced anatomy\
        and physiology, clinical pharmacology of anaesthetics and adjunctive drugs, pain\
        management, acute care and operating room practice, clinical technology and\
        procedures, emergency intervention, patient assessment and education, and legal\
        issues.\
        Provide prenatal care to\
        pregnant women and to mothers and their newborn infants and to prepare\
        registered nurses to independently deliver babies. Includes instruction in predelivery screening, physician referral, and the care of infants during the delivery\
        and immediate post-delivery phases. Includes instruction in perinatal and\
        newborn health assessment, stabilisation and care, pathophysiology of\
        pregnancy, foetuses and the newborn, clinical management of high-risk\
        pregnancies and newborns, perinatal and neonatal technology and clinical\
        procedures, and patient education.\
        A research area of study that focuses on the study of advanced clinical practices,\
        research methodologies, the administration of complex nursing services, and that\
        prepares nurses to further the progress of nursing research through\
        experimentation and clinical applications.\
        provide care for children\
        from infancy through adolescence. Includes instruction in the administration of\
        medication and treatments, assisting physicians, patient examination and referral,\
        and planning and delivery health maintenance and head education programmes.\
        promote mental health and\
        provide nursing care to patients with mental, emotional or behavioural disorders in\
        mental institutions or other settings. Includes instruction in psychopathology,\
        behavioural heal, counselling and intervention strategies, psycho education,\
        mental health assessment and dual diagnosis, stabilisation and management of\
        psychotic illness, psychiatric care and rehabilitation, substance abuse and crisis\
        intervention.\
        Promote health and provide\
        preventive and curative nursing services for groups or communities under the\
        supervision of a public health agency. Includes instruction in community and rural\
        health, family therapy and intervention, disease prevention, health education, and\
        community health assessment.\
        Provide care to patients\
        before, during and after surgery, and to provide tableside assistance to surgeons.\
        Includes instruction in operating room safety and preparation, aseptic technique,\
        anaesthesia, patient preparation, surgical instruments and procedures,\
        sterilisation and disinfecting, surgical drugs and solutions, haemostasis,\
        emergency procedures, and patient/family education.\
        deliver direct patient and\
        client care in clinical settings. Includes instruction in clinical\
        pharmacotherapeutics, advance clinical practice, holistic nursing, nursing practice\
        and health care policy, administration and consultation services, health\
        assessment, patient stabilisation and care, and patient education.\
        Provide specialised care to\
        patients with life-threatening problems, including those in intensive care facilities,\
        hospital emergency units and on life support. Includes instruction in adult,\
        neonatal and paediatric critical care; technical skills; patient monitoring and\
        assessment; normal and abnormal readings, and troubleshooting.\
        Deliver nursing health care\
        services to workers and worker populations in clinical settings and at job sites.\
        Includes instruction in public and community health; occupational safety and\
        health; occupational health surveillance; case management; fitness for duty\
        testing; medication; allergies and immunisation; emergency and ambulatory care;\
        and applicable laws and regulations.',

    1101: 'The descriptive, historical and theoretical\
        aspects of language, its nature, structure, varieties and development, including\
        especially the sound system (phonology), grammatical system (morphology and\
        syntax), lexical system (vocabulary and semiology), and writing system.\
        two or more literary traditions in the original\
        languages or in translation. Includes instruction in comparative linguistics,\
        applicable indigenous and foreign languages and literature, literary criticism, and\
        applications to genre, period, national and textural studies as well as literary forms\
        such as poetry, prose and drama.\
        human interpersonal communication from the\
        scientific/behavioural and humanistic perspectives. Includes instruction in the\
        theory and physiology of speech, the history of discourse, the structure and\
        analysis of argument and types of public speech, the social role of speech, oral\
        interpretation of literature, interpersonal interactions, and the relation of speech to\
        nonverbal and other forms of message exchanges.\
        language interpretation and translation,\
        document design, copywriting, text editing, lexicography, terminology, language\
        management and (natural) language technology.',

    1102: 'English language, including its history,\
        structure and related communications skills; and the literature and culture of\
        English-speaking peoples.\
        principles of English vocabulary, grammar,\
        morphology, syntax and semantics, as well as techniques of selecting,\
        developing, arranging, combining and expressing ideas in appropriate written\
        forms.\
        Process and techniques of original English\
        composition in various literary forms such as the short story, poetry, the novel,\
        and others. Includes instruction in technical and editorial skills, criticism, and the\
        marketing of finished manuscripts.\
        literatures and literary developments of\
        mother tongue English speakers, from the origins of the English language to the\
        present. Includes instruction in period and genre studies, author studies, country\
        and regional specialisations, literary criticism, and the study of folkloric traditions.\
        theory, methods and skills needed for\
        writing and editing scientific, technical and business papers and monographs in\
        English.',

    1302: 'The chemistry of living\
        systems, their fundamental chemical substances and reactions, and their\
        chemical pathways and information transfer systems, with particular reference to\
        carbohydrates, proteins, lipids and nucleic acids. Includes instruction in biologyorganic chemistry, protein chemistry, bioanalytical chemistry, bioseparations,\
        regulatory biochemistry, enzymology, hormonal chemistry, calorimetry, and\
        research methods and equipment operation.\
        the application of physics principles to the\
        scientific study the mechanisms of biological processes and assemblies at all\
        levels of complexity. Includes instruction in research methods and equipment\
        operation and applications to subjects such as bioenergetics, biophysical theory\
        and modelling, electrophysics, membrane biology, channels, receptors and\
        transporters, contractility and muscle function, protein shaping and folding,\
        molecular and supramolecular structures and assemblies, and computational\
        science.\
        The structure and function\
        of biological macromolecules and the role of molecular constituents and\
        mechanisms in supramolecular assemblies and cells. Includes instruction in such\
        topics as molecular signalling and transduction, regulation of cell growth, enzyme\
        substrates and mechanisms of enzyme action, DNA-protein interaction, and\
        applications to fields such as biotechnology, genetics, cell biology and physiology.\
        the scientific relationship of physiological\
        Function to the structure and actions of macromolecules and supramolecular\
        assemblies such as multienzyme complexes, membranes and viruses. Includes\
        instruction in the chemical mechanisms of regulation and catalysis, protein\
        synthesis and other syntheses, and biomolecular chemical reactions.\
        the dynamics and interactions of\
        Macromolecules and other three-dimensional ultrastructures, the architecture of\
        supramolecular structures, and energy transfer in biomolecular systems. Includes\
        instruction in energy transduction, structural dynamics, mechanisms of electron\
        and proton transfer in biological systems, bioinformatics, automated analysis, and\
        specialised research techniques.\
        submolecular and\
        Molecular components and assemblies of living systems and how they are\
        organised into functional units such as cells and anatomic tissues. Includes\
        instruction in glycoprotein, carbohydrate, protein and nucleic acid structures and\
        chemistry; cytoskeletal structure; nuclear and intracellular structures; molecular\
        recognition; molecular chaperones; transcription and folding; multicellular\
        organisation; microtubules and microfilaments; cell differentiation;\
        immunophysics; and DNA sequencing.\
        the effects of light energy\
        on living organisms, the manufacture and processing of luminescence by\
        organisms, and the uses of light in biological research. Includes instruction in\
        bioluminescence, chronobiology, photomedicine, environmental photobiology,\
        organic photochemistry, photomorphogenesis, photoreceptors and\
        photosensitisation, molecular mechanics of photosynthesis, phototechnology,\
        vision, ultraviolet radiation, radiation physics, and spectral research methods.\
        the effects of radiation on organisms and\
        biological systems. Includes instruction in particle physics, ionisation, and\
        biophysics of radiation perturbations, cellular and organismic repair systems,\
        genetic and pathological effects of radiation, and the measurement of radiation\
        dosages.',

    1303: 'Plants, related microbial\
        and algal organisms, and plant habitats and ecosystem relations. Includes\
        instruction in plant anatomy and structure, phytochemistry, cytology, plant\
        genetics, plant morphology and physiology, plant ecology, plant taxonomy and\
        systematics, paleobotany, and applications of biophysics and molecular biology.\
        plant diseases and plant\
        Health, and the development of disease control mechanisms. Includes instruction\
        in plant anatomy and physiology; pathogenesis; molecular plant virology;\
        molecular genetics; bacterial epidemiology; causal agent identification; host/agent\
        interactions; disease resistance and response mechanisms; developing plant\
        disease treatments; disease prevention; and disease physiology and control.\
        plant internal dynamics\
        and systems, plant-environmental interaction, and plant life cycles and processes.\
        Includes instruction in cell and molecular biology, plant nutrition, plant respiration,\
        plant growth, behaviour and reproduction, photosynthesis, plant systematics, and\
        ecology.\
        Molecular biology,\
        biochemistry, and biophysics to the study of biomolecular structures, functions\
        and processes specific to plants and plant substances. Includes instruction in the\
        biochemistry of plant cells, nuclear-cytoplasmic interactions, molecular\
        cytostructures, photosynthesis, plant molecular genetics, and the molecular\
        biology of plant diseases.',

    1306: 'The biology of\
        animal species and phyla, with reference to their molecular and cellular systems,\
        anatomy, physiology, and behaviour. Includes instruction in molecular and cell\
        biology, microbiology, anatomy and physiology, ecology and behaviour,\
        evolutional biology, and applications to specific species and phyla.\
        insect species and\
        populations in respect of their life cycles, morphology, genetics, physiology,\
        ecology, taxonomy, population dynamics, and environmental and economic\
        impacts. Includes instruction in applicable biological and physical sciences as\
        well as insect toxicology and the biochemical control of insect populations.\
        function, morphology,\
        Regulation, and intercellular communications and dynamics within vertebrate and\
        invertebrate in animal species, with comparative applications to homo sapiens\
        and its relatives and antecedents. Includes instruction in reproduction, growth,\
        lactation, digestion, performance, behavioural adaptation, sensory perception,\
        motor action, phylogenetics, biotic and abiotic function, membrane biology, and\
        related aspects of biochemistry and biophysics.\
        the psychological and\
        Neurological bases of animal sensation, perception, cognition, behaviour and\
        behavioural interactions within and outside the species. Includes instruction in\
        ethology, behavioural neuroscience, neurobiology, behavioural evolution,\
        cognition and sensory perception, motivators, learning and instinct, hormonal\
        controls, reproductive and developmental biology, community ecology, functional\
        behaviour, and applications to specific behaviours and patterns as well as to\
        specific phyla and species.\
        biological principles to the\
        Vertebrate wildlife, wildlife habitats, and related ecosystems in remote\
        and urban areas. Includes instruction in animal ecology, adaptational biology,\
        urban ecosystems, natural and artificial habitat management, limnology, wildlife\
        pathology, and vertebrate zoological specialisations such mammalogy,\
        herpetology, ichthyology, ornithology, and others.',

    1404: 'Composition and\
        behaviour of matter, including its micro- and macrostructure, the processes of\
        chemical change, and the theoretical description and laboratory simulation of\
        these phenomena.\
        techniques for analysing\
        and describing matter, including its precise composition and the interrelationships\
        of constituent elements and compounds. Includes instruction in spectroscopy,\
        chromatography atomic absorption, photometry, chemical modelling,\
        mathematical analysis, laboratory analysis procedures and equipment\
        maintenance, and applications to specific research, industrial and health\
        problems.\
        The elements and their\
        compounds, other than the hydrocarbons and their derivatives. Includes\
        instruction in the characterisation and synthesis of non-carbon molecules,\
        including their structure and their bonding, conductivity, and reactive properties;\
        research techniques such as spectroscopy, X-ray diffraction, and photoelectron\
        analysis; and the study of specific compounds, such as transition metals, and\
        compounds composed of inorganic and organic molecules.\
        the properties and\
        behaviour of hydrocarbon compounds and their derivatives. Includes instruction\
        in molecular conversion and synthesis, the molecular structure of living cells and\
        systems, the mutual reactivity of organic and inorganic compounds in\
        combination, the spectroscopic analysis of hydrocarbon compounds, and\
        applications to specific problems in research, industry and health.\
        the theoretical properties\
        of matter, and the relation of physical forces and phenomena to the chemical\
        structure and behaviour of molecules and other compounds. Includes instruction\
        in reaction theory, calculation of potential molecular properties and behaviour,\
        computer simulation of structures and actions, transition theory, statistical\
        mechanics, phase studies, quantum chemistry, and the study of surface\
        properties.\
        Synthesised\
        macromolecules and their interactions with other substances. Includes instruction\
        in molecular bonding theory, polymerisation, properties and behaviour of unstable\
        compounds, the development of tailored polymers, transition phenomena, and\
        applications to specific industrial problems and technologies.\
        structural phenomena\
        combining the disciplines of physical chemistry and atomic/molecular physics.\
        Includes instruction in heterogeneous structures, alignment and surface\
        phenomena, quantum theory, mathematical physics, statistical and classical\
        mechanics, chemical kinetics, liquid crystals and membranes, molecular synthesis\
        and design, and laser physics.', 

    1405: 'The spatial distribution\
        and interrelationships of people, natural resources, plant and animal life. Includes\
        instruction in historical and political geography, cultural geography, economic and\
        physical geography, regional science, cartographic methods, remote sensing,\
        spatial analysis, and applications to areas such as land-use planning,\
        development studies, and analyses of specific countries, regions, and resources.\
        map making and the\
        application of mathematical, computer, and other techniques to the science of\
        mapping geographic information. Includes instruction in cartographic theory and\
        map projections, computer-assisted cartography, map design and layout,\
        photogrammetry, air photo interpretation, remote sensing, cartographic editing,\
        and applications to specific industrial, commercial, research, and governmental\
        mapping problems.\
        Biological, chemical, and\
        physical principles to the study of the physical environment and the solution of\
        environmental problems, including subjects such as abating or controlling\
        environmental pollution and degradation; the interaction between human society\
        and the natural environment; and natural resources management. Includes\
        instruction in biology, chemistry, physics, geosciences, climatology, statistics, and\
        mathematical modelling.\
        Plan, develop, manage, and\
        evaluate programmes to protect natural habitats and renewable natural resources.\
        Includes instruction in the principles of wildlife and conservation biology,\
        environmental science, animal population surveying, natural resource economics,\
        management techniques for various habitats, applicable law and policy,\
        administrative and communications skills, and public relations.', 

    1406: 'The earth; the forces\
        acting upon it; and the behaviour of the solids, liquids and gases comprising it.\
        Includes instruction in historical geology, geomorphology and sedimentology, the\
        chemistry of rocks and soils, stratigraphy, mineralogy, petrology, geostatistics,\
        volcanology, glaciology, geophysical principles, and applications to research and\
        industrial problems.\
        The chemical properties\
        and behaviour of the silicates and other substances forming, and formed by\
        geomorphological processes of the earth and other planets. Includes instruction\
        in chemical thermodynamics, equilibrium in silicate systems, atomic bonding,\
        isotopic fractionation, geochemical modelling, specimen analysis, and studies of\
        specific organic and inorganic substances.\
        the physics of solids and\
        its application to the study of the earth and other planets. Includes instruction in\
        gravimetric, seismology, earthquake forecasting, magnetrometry, electrical\
        properties of solid bodies, plate tectonics, active deformation, thermodynamics,\
        remote sensing, geodesy, and laboratory simulations of geological processes.\
        extinct life forms and\
        associated fossil remains, and the reconstruction and analysis of ancient life\
        forms, ecosystems and geologic processes. Includes instruction in sedimentation\
        and fossilisation processes, fossil chemistry, evolutionary biology, paleoecology,\
        paleoclimatology, trace fossils, micropaleontology, invertebrate palaeontology,\
        vertebrate palaeontology, paleobotany, field research methods, and laboratory\
        research and conservation methods.\
        the occurrence,\
        circulation, distribution, chemical and physical properties, and environmental\
        interaction of surface and subsurface waters, including groundwater. Includes\
        instruction in geophysics, thermodynamics, fluid mechanics, chemical physics,\
        geomorphology, mathematical modelling, hydrologic analysis, continental water\
        processes, global water balance, and environmental science.\
        the igneous,\
        Metamorphic, and hydrothermal processes within the earth and the mineral, fluid,\
        rock and ore deposits resulting from them. Includes instruction in mineralogy,\
        crystallography, petrology, volcanology, economic geology, meteoritics,\
        geochemical reactions, deposition, compound transformation, core studies,\
        theoretical geochemistry, computer applications, and laboratory studies.\
        the chemical\
        components, mechanisms, structure, and movement of ocean waters and their\
        interaction with terrestrial and atmospheric phenomena. Includes instruction in\
        material inputs and outputs, chemical and biochemical transformations in marine\
        systems, equilibrium studies, inorganic and organic ocean chemistry,\
        oceanographic processes, sediment transport, zone processes, circulation,\
        mixing, tidal movements, wave properties, and seawater properties.',

    1407: 'Matter and energy,\
        and the formulation and testing of the laws governing the behaviour of the matterenergy continuum. Includes instruction in classical and modern physics,\
        electricity and magnetism, thermodynamics, mechanics, wave properties, nuclear\
        processes, relativity and quantum theory, quantitative methods, and laboratory\
        methods.\
        The behaviour of matterenergy phenomena at the level of atoms and molecules. Includes instruction in\
        chemical physics, atomic forces and structure, fission reactions, molecular orbital\
        theory, magnetic resonance, molecular bonding, phase equilibria, quantum theory\
        of solids, and applications to the study of specific elements and higher\
        compounds.\
        The basic constituents of\
        sub-atomic matter and energy, and the forces governing fundamental processes.\
        Includes instruction in quantum theory, field theory, single-particle systems,\
        perturbation and scattering theory, matter-radiation interaction, symmetry, quarks,\
        capture, Schroedinger mechanics, methods for detecting particle emission and\
        absorption, and research equipment operation and maintenance.\
        properties and behaviour\
        of matter at high temperatures, such that molecular and atomic structures are in a\
        disassociated ionic or electronic state. Includes instruction in\
        magnetohydrodynamics, free electron phenomena, fusion theory, electromagnetic\
        fields and dynamics, plasma and non-linear wave theory, instability theory,\
        plasma shock phenomena, quantitative modelling, and research equipment\
        operation and maintenance.\
        the properties and\
        Behaviour of atomic nuclei instruction in nuclear reaction theory, quantum\
        mechanics, energy conservation, nuclear fission and fusion, strong and weak\
        atomic forces, nuclear modelling, nuclear decay, nucleon scattering, pairing,\
        photon and electron reactions, statistical methods, and research equipment\
        operation and maintenance.\
        light energy, including its\
        structure, properties and behaviour under different conditions. Includes\
        instruction in wave theory, wave mechanics, electromagnetic theory, physical\
        optics, geometric optics, quantum theory of light, photon detecting, laser theory,\
        wall and beam properties, chaotic light, non-linear optics, harmonic generation,\
        optical systems theory, and applications to engineering problems.\
        solids and related states\
        of matter at low energy levels, including liquids and dense gases. Includes\
        instruction in statistical mechanics, quantum theory of solids, many-body theory,\
        low temperature phenomena, electron theory of metals, band theory, crystalline\
        structures, magnetism and superconductivity, equilibria and dynamics of liquids,\
        film and surface phenomena, quantitative modelling, and research equipment\
        operation and maintenance.\
        sound, and the\
        properties and behaviour of acoustic wave phenomena under different conditions.\
        Includes instruction in wave theory, the acoustic wave equation, energy\
        transformation, vibration phenomena, sound reflection and transmission,\
        scattering and surface wave phenomena, singularity expansion theory, ducting,\
        and applications to specific research problems such as underwater acoustics,\
        crystallography and health diagnostics.\
        scientific and mathematical formulation and\
        evaluation of the physical laws governing, and models describing, matter-energy\
        physics, and the analysis of related experimental designs and results. Includes\
        instruction in classical and quantum theory, computational physics, relativity\
        theory, field theory, mathematics of infinite series, vector and coordinate analysis,\
        wave and particle theory, advanced applied calculus and geometry, analyses of\
        continuum, cosmology, and statistical theory and analysis.',

    1702: 'Religious belief and\
        specific religious and quasi-religious systems. Includes instruction in\
        phenomenology; the sociology, psychology, philosophy, anthropology, literature\
        and art of religion; mythology; scriptural and textual studies; religious history and\
        politics.', 

    1703: 'The cultural, religious and spiritual\
        phenomena indigenous to the continent of Africa - such as African conceptions of\
        God(s) and the cosmos.\
        the philosophy preached by Siddartha\
        Gautama, the Buddha, in ancient India and subsequently interpreted by his\
        followers and evolving into a religion; together with the intellectual, cultural, social,\
        and ritual developments of the faith and its branches. Includes instruction in\
        Buddhist sacred literature (Tipitaka, etc.) and study of one or more of the main\
        branches including Early Buddhism, Hinayana, Theravada, Madhyamaka,\
        Yogacara, Pure Land, Shingon, Tendai, Nichiren Shu, Zen, Tibetan, Chinese,\
        Korean, Vietnamese, and others.\
        Christian theology focuses on the foundational documents, of the\
        history and development, and of the faith, ethics and practices of the Christian\
        church and tradition, and therefore includes scholarly disciplines like systematic\
        theology/dogmatics, practical/pastoral theology (with its sub-disciplines),\
        contextual theologies, theological ethics, hermeneutics, missiology, ecumenics,\
        church law, church history, and Biblical studies (with its sub-disciplines Hebrew\
        Bible/Old Testament and New Testament, making use of languages like Hebrew,\
        Aramaic and Greek).\
        The group of South Asian theologies and\
        philosophies collectively known as Hinduism; together with the religious history\
        and cultural and social manifestations of the faith. Includes instruction in Hindu\
        theology and philosophy (dharma); literature (Vedas, Upanishads, Epics, and\
        commentaries); the Hindu Pantheon, sects, and movements; schools and\
        disciplines; and related arts and sciences.\
        Islam as preached by the Prophet Muhammad\
        in 7th century Arabia and subsequently interpreted and elaborated by Muslim\
        scholars and others; together with the cultural and social milieu related to the faith\
        in various periods, localities and branches. Includes instruction in Muslim scripture\
        and related written authorities and commentaries (Quran, Tafsir, Hadith/Sunnah,\
        Sirah); Islamic law and jurisprudence; the various branches of Muslim theology\
        including Sunni, Shia, Sufism and others; as well as the development of the\
        religion of Islam and Muslim society from the beginnings to the present.\
        the history, culture, and religion of the Jewish\
        people. Includes instruction in Jewish religious heritage, sacred texts, and law;\
        Jewish philosophy and intellectual history; Jewish life and culture, both in Israel\
        and the Jewish Diaspora; historical Jewish minority cultures such as the Yiddish,\
        Sephardic, and other; anti-Semitism, gentile relations and Shoa; Zionism; and\
        modern developments within Judaism. ',
        
    1808: 'Behaviour of individuals in the roles of teacher and learner, the nature and\
        effects of learning environments, and the psychological effects of methods,\
        resources, organisation and non-school experience on the educational process.\
        Includes instruction in learning theory, human growth and development, research\
        methods and psychological evaluation.\
        Clinical and counselling\
        psychology principles to the diagnosis and treatment of student behavioural\
        problems. Includes instruction in child and/or adolescent development; learning\
        theory; testing, observation and other procedures for assessing educational,\
        personality, intelligence and motorskill development; therapeutic intervention\
        strategies for students and families; identification and classification of disabilities\
        and disorders affecting learning; school psychological services planning;\
        supervised counselling practice; ethical standards; and applicable regulations.',

    1814: 'Individual and group\
        behaviour in organisational settings and the physical environmental context within\
        which productive behaviour occurs; the application of the paradigms, theories,\
        models, constructs and principles of psychology to issues related to the world of\
        work in order to enhance the effectiveness of individual, group and organisational\
        behaviour and the well-being of the participants.\
        individual differences, developmental\
        challenges, career models, career choice, entry into the world of work, career\
        development, career stages, occupational challenges, and management and\
        support systems.\
        Intra-individual variables, like personality traits,\
        perceptions, attitudes, values, and motivation which affect organisational\
        behaviour; interpersonal, group and intergroup behaviours, like leadership, power\
        relations, decision-making and communication; organisational structure and\
        design, organisational culture, organisational change and development.\
        individual differences in health and well-being\
        in organisations, improving worker health through interventions, group/team and\
        normative influences on health and well-being, antisocial work behaviour and\
        organisational health, employee assistance programmes, workplace health\
        promotion, integration and future directions, the management of HIV/AIDS in the\
        workplace. Why people fail in their jobs, a systematic performance analysis\
        approach to the solution of ineffective performance, the development of taxonomy\
        of work-related performance dysfunctions, broad areas of work dysfunction,\
        dysfunctional working conditions, diagnosis of work-related dysfunctions,\
        prevention and management of performance dysfunctions.',

    2003: 'Interpretation of the\
        past, including the gathering, recording, synthesising and criticising of evidence\
        and theories about past events. Includes instruction in historiography; historical\
        research methods; studies of specific periods, issues and cultures; and\
        applications to areas such as historic preservation, public policy, and records\
        administration.\
        The development, changes, past events,\
        discoveries, trends, individuals, institutions, ideas, artefacts, and the systematic\
        accounting of other phenomena associated with the continent of Africa and its\
        inhabitants, but excluding the detailed history of South Africa and its inhabitants.\
        the development of American society, culture,\
        and institutions from the Pre-Columbian period to the present. Includes instruction\
        in American historiography, American history sources and materials, historical\
        research methods, and applications to the study of specific themes, issues,\
        periods, and institutions.\
        The development of the societies, cultures, and\
        institutions of the Asian Continent from their origins to the present. Includes\
        instruction in the historiography of specific cultures and periods; sources and\
        materials; historical research methods; and applications to the study of specific\
        themes, issues, periods, and institutions.\
        the development of European society, culture,\
        and institutions from the origins to the present. Includes instruction in European\
        historiography, European history sources and materials, historical research\
        methods, and applications to the study of specific themes, issues, periods, and\
        institutions.\
        The historical evolution of scientific theories\
        and science applications and technologies, as well as the philosophy of science\
        and its historical socio-economic context. Includes instruction in the construction\
        and methods of philosophical inquiry, historiography of science research methods\
        in the history of the scientific and engineering disciplines, including mathematics.\
        The detailed study of the development, changes, past events, discoveries, trends,\
        individuals, institutions, ideas, artefacts, and the systematic accounting of other\
        phenomena associated with South Africa and its inhabitants.\
        the history of the ancient cultures of the Near\
        East; an overview of the history, culture and political development of Ancient\
        Greece; the social history of Ancient Rome; the development of the Roman legal\
        system, constitutional development in Rome, basic Latin terms (legal and other) in\
        general use today; and an overview of the history of other ancient cultures.',
    
    2006: 'Political institutions and\
        behaviour. Includes instruction in political philosophy, political theory, comparative\
        government and politics, political parties and interest groups, public opinion,\
        political research methods, studies of the government and politics of specific\
        countries, and studies of specific political institutions and processes.\
        Comparative analysis of the\
        similarities and differences of political institutions, processes, and behaviour in\
        different countries and political subdivisions.\
        the study of economic, sociological, political,\
        legal, cultural, and other factors which influence the present relations between\
        nations.\
        International politics\
        and institutions, and the conduct of diplomacy and foreign policy. Includes\
        instruction in international relations theory, foreign policy analysis, national\
        security and strategic studies, international law and organisation, the comparative\
        study of specific countries and regions, and the theory and practice of diplomacy.',
    
    2007: 'The systematic study of human social\
        institutions and social relationships. Includes instruction in industrial sociology,\
        social theory, sociological research methods, social organisation and structure,\
        social stratification and hierarchies, dynamics of social change, family structures,\
        social deviance and control, and applications to the study of specific social\
        groups, social institutions, and social problems. Focuses also on the human\
        impact on the environment and the individual and group behaviours with respect\
        to health and illness.\
        The systematic examination of population\
        models and population phenomena, and related problems of social structure and\
        behaviour. Includes instruction in population growth, spatial distribution, mortality\
        and fertility factors, migration and its effects, urbanisation, dynamic population\
        modelling, population estimation and projection, mathematical and statistical\
        analysis of population data, population policy studies, and applications to\
        problems in economics and government planning.\
        developing societies. Issues such as develop-\
        ment, underdevelopment, poverty and global stratification are addressed.',
        
    2008: 'The professional practice of social\
        welfare administration and counselling, and that focuses on the study of organised\
        means of providing basic support services for vulnerable individuals, groups and\
        communities. Includes instruction in social welfare policy; case work planning;\
        social counselling and intervention strategies; administrative procedures and\
        regulations; and specific applications in areas such as child welfare and family\
        services, probation, employment services, and disability counselling.\
        Plan, manage, and implement\
        social services for children, youth, and families. Includes instruction in child\
        development and psychology, adolescence, family studies, social work, social\
        services administration, juvenile and family law, programme and facilities\
        planning, youth leadership, counselling, probation, casework, applicable\
        procedures, and regulations, and professional standards and ethics.'
    }

    return np.array(list(descriptions.values()))

