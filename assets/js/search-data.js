// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-publications",
          title: "publications",
          description: "A comprehensive collection of academic publications organized by type and sorted chronologically.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/publications/";
          },
        },{id: "nav-repositories",
          title: "repositories",
          description: "A showcase of my GitHub repositories and contributions.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/repositories/";
          },
        },{id: "nav-cv",
          title: "cv",
          description: "A comprehensive overview of my academic background, research experience, and professional achievements.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "dropdown-bookshelf",
              title: "bookshelf",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "/books/";
              },
            },{id: "post-notes-for-ajoint-methods",
        
          title: "Notes for Ajoint Methods",
        
        description: "The adjoint method, introduced in the 2018 neural ODE paper, provides an efficient approach for computing gradients in neural ODEs. This article presents a detailed mathematical derivation of the adjoint method&#39;s core formulas and their applications.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2024/ajoint-methods/";
          
        },
      },{id: "post-notes-for-diffusion-models",
        
          title: "Notes for Diffusion Models",
        
        description: "Diffusion models are an important class of modern generative models. This article provides a brief introduction to distribution-based generative algorithms such as diffusion, SDE, and consistency models based on my personal understanding.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2024/diffusion-models/";
          
        },
      },{id: "books-a-brief-history-of-chinese-philosophy",
          title: 'A Brief History of Chinese Philosophy',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/chinese_philosophy/";
            },},{id: "books-a-brief-history-of-chinese-philosophy",
          title: 'A Brief History of Chinese Philosophy',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/reussell_philosophy/";
            },},{id: "books-introduction-to-stochastic-processes",
          title: 'Introduction to Stochastic Processes',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/stochastical_process/";
            },},{id: "books-introduction-to-smooth-manifolds",
          title: 'Introduction to Smooth Manifolds',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/smooth_manifolds/";
            },},{id: "news-i-m-excited-to-share-two-new-articles-on-my-blog-notes-for-diffusion-models-and-notes-for-adjoint-methods-these-articles-are-currently-being-refined-to-ensure-accuracy-and-clarity",
          title: 'I’m excited to share two new articles on my blog: Notes for Diffusion...',
          description: "",
          section: "News",},{id: "news-release-of-alphachimp-tracking-and-behavior-recognition-of-chimpanzees",
          title: 'Release of AlphaChimp: Tracking and Behavior Recognition of Chimpanzees.',
          description: "",
          section: "News",},{id: "news-alphachimp-is-accepted-by-cv4animals-cvpr-2025-workshop",
          title: 'AlphaChimp is accepted by CV4Animals CVPR 2025 Workshop.',
          description: "",
          section: "News",},{id: "news-release-of-clone-holistic-closed-loop-whole-body-teleoperation-for-long-horizon-humanoid-control",
          title: 'Release of CLONE: Holistic Closed-Loop Whole-Body Teleoperation for Long-Horizon Humanoid Control.',
          description: "",
          section: "News",},{id: "news-clone-is-accepted-by-corl-2025",
          title: '🔥 CLONE is accepted by CoRL 2025.',
          description: "",
          section: "News",},{id: "news-release-of-cola-learning-human-humanoid-coordination-for-collaborative-object-carrying",
          title: 'Release of COLA: Learning Human-Humanoid Coordination for Collaborative Object Carrying.',
          description: "",
          section: "News",},{id: "news-cola-is-accepted-by-icra-2026",
          title: '🔥 COLA is accepted by ICRA 2026.',
          description: "",
          section: "News",},{id: "news-release-of-lessmimic-long-horizon-humanoid-interaction-with-unified-distance-field-representations-check-out-the-arxiv-paper-and-code",
          title: 'Release of LessMimic: Long-Horizon Humanoid Interaction with Unified Distance Field Representations. Check out...',
          description: "",
          section: "News",},{id: "news-release-of-omniclone-engineering-a-robust-all-rounder-whole-body-humanoid-teleoperation-system",
          title: 'Release of OmniClone: Engineering a Robust, All-Rounder Whole-Body Humanoid Teleoperation System.',
          description: "",
          section: "News",},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%79%75%74%61%6E%67.%6C%69%6E@%73%74%75.%70%6B%75.%65%64%75.%63%6E", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/Yutang-Lin", "_blank");
        },
      },{
        id: 'social-orcid',
        title: 'ORCID',
        section: 'Socials',
        handler: () => {
          window.open("https://orcid.org/0009-0004-4933-1203", "_blank");
        },
      },{
        id: 'social-wechat_qr',
        title: 'Wechat_qr',
        section: 'Socials',
        handler: () => {
          window.open("", "_blank");
        },
      },{
        id: 'social-dblp',
        title: 'DBLP',
        section: 'Socials',
        handler: () => {
          window.open("https://dblp.uni-trier.de/pid/165/0916-1.html", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=3dekOUcAAAAJ", "_blank");
        },
      },{
        id: 'social-custom_social',
        title: 'Custom_social',
        section: 'Socials',
        handler: () => {
          window.open("https://www.xiaohongshu.com/user/profile/5fffb012000000000101ed53", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
