---
layout: page
permalink: /publications/
title: publications
description: A comprehensive collection of academic publications organized by type and sorted chronologically.
nav: true
nav_order: 2
---

<!-- _pages/publications.md -->

<!-- Bibsearch Feature -->

{% include bib_search.liquid %}

<div class="publications">

 {% bibliography --template bib --group_by type --group_order ascending %}

</div>
