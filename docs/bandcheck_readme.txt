Bandcheck.py hjälper till att sätta rätt värden på limits i settings filen för hdim2.py.

Arbetsflöde:
1. Öppna Pylon och spela in en en (.avi) film då det tomma bandet rör på sig.
2. Ta några bilder (ca fem) på diskar.
3. Lägg filerna (bilder och video) i en mapp under susmag.
4. Kör colorcheck.py <mappnamn>
5. I histogrammet för videon (på det tomma bandet), hitta min och max värden för varje lager. 
6. I histogrammen för bilderna på diskarne, hitta i vika lager dom skiljer sig åt. 
7. Lägg in min max värden för videon för lagren som skiljer sig från bilderna i settingsfilen.
8. Testa. 
Om det inte funkar bra:
    * Kör lagrena en och en genom att sätta min och max till 0. 
    * Flytta min och max neråt resp. uppåt för att göra området lite större. 

