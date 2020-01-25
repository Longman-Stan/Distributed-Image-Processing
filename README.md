# Distributed image processing
			Lungu-Stan Vlad-Constantin, 334CB, tema3 APD
	
  	Abordarea mea este una simpla, dar, zic eu, eficienta. In primul rand imi declar filtrele
deja rotite, global, fiind constante. Tot global imi declar alte variabile necesare, cum ar fi 
datele despre imagine( width, height, si numarul de canale, tipul de imagine (P5 sau P6)), 
imaginea in sine, base_size si num_add (variabile folosite pentru calcularea portiunii de imagine
care ii revine fiecarui proces),indicii extremitatilor portiunii de imagine pentru procesul curent
si my_part, un buffer de dimensiuea portiunii pe care trebuie sa o proceseze procesul in cauza. 
Il declar global pentru ca il aloc in main(pentru a nu aloca de fiecare data cand aplic un filtru, 
ar fi ineficient).
	Dupa initializarea MPI, apelez functia init_data(nume_imagine), care:
	1. Imparte filtrele la numitorii corespunzatori
	2. deschide si citeste imaginea
	Imaginea o tin in memorie ca o zona continua de octeti(unsigned char), de marimea (width+2)*
(height+2)*nrchannels. Bordarea o fac cu niste formule deloc dragute, dar care merg(acum ca ma 
uit puteam sa nu bordez, pentru ca am initializat cu calloc...but hey....it was good work).
	Dupa citirea imaginii, imi calculez portiunea de imagine pe care trebuie sa o procesez(dupa
formulele pe care le-am dedus in primul laborator, am ales sa le folosesc pe acestea in detrimentul
celor din curs pentru ca sunt ale mele) si imi aloc o zona de memorie corespunzatoare.
	Acum trec la treaba. Parcurg lista de filtre, iar pentru fiecare in parte fac mai multi pasi:
	1. aplic filtrul prin functia apply_filter(nume_filtru,st_interval_de_procesat, dr_interval_de_
		procesat)
	2. trimit portiunea de imagine pe care am procesat-o procesului parinte(0), care combina rezultatul
		aplicarii filtrului
	3. daca nu e ultimul filtru, primesc imaginea completa de la parinte(ca sa fie o calculare corecta
		a rezultatului)((nu primesc imaginea si in cazul aplicarii ultimului filtru pentru ca ar fi 
		degeaba, fiind pasul final))
	
	Dupa ce treaba e gata, procesul parinte scrie imaginea in fisierul al carui nume este dat de catre
al doilea argument al programului.
	In program pot fi gasite doua functii, write_image_binary, care scrie imaginea in format binar si
write_image_ascii, care scrie imaginea in format ascii(a fost util pentru debugging, la inceput).
	filter_Pixel este functia care aplica filtrul "kernel" unui pixel din imagine. Pixelul este dat ca
adresa de memorie.
	apply_filter() primeste numele unui filtru si extremitatile intervalului de care e raspunzator un
proces si aplica filtrul fiecarui pixel din acea zona de memorie.
	send_image() e functia de adunare a bucatilor de imagine obtinute prin delegarea responsabilitatii
celorlalte procese. Procesul parinte primeste, pe rand, de la celelalte procese bucata lor de imagine pe
care au aplicat un filtru si o pune peste imaginea pe care o detine. Dupa ce a primit toate mesajele,
acesta are imaginea corect filtrata. Toate celelalte procese isi trimit partea lor de imagine.
	recv_image() e o functie pe care am facut-o uitand cu desavarsire de MPI_broadcast. Procesul parinte
trimite imaginea filtrata pe care a compus-o prin functia precedenta tuturor celorlalte procese, pentru a
le permite sa aplice corect un nou filtru.
	
	Astfel, programul e gata. Si face si ce trebuie. Acesta da exact imaginile de output de referinta,
verificare fiind facuta cu diff. Pentru a face mai usor procesul de verificare a imaginilor am facut un 
script, "checker.sh", care primeste 3 argumente:
	1. Numele directorului care contine directoarele PNM si PNG,cu imaginile de input
	2. Numele executabilului
	3. Numele directorului care contine directoarele pnm si png,cu imaginile de referinta
	Scriptul ruleaza programul dat prin al doilea argument pentru toate combinatiile de imagini de
intrare si filtre(atat cele simple, cat si bssembssem), folosind 4 procese. Dupa aceea compara
rezultatul rularii programului cu imaginea de referinta corespunzatoare, cu diff, si, in cazul in 
care imaginle sunt identice, afiseaza mesajul "output correct".
	Dupa aceea ruleaza programul pentru 1, 2, respectiv 3 procese si compara imaginile rezultate
(1 cu 2, 2 cu 3 si 3 cu output, rezultatul rularii care verifica corectitudinea). Daca toate sunt,
identice, afiseaza mesajul "the output is the same, no matter the number of processes", indicand
o functionare corecta a programului. La final sterge fisierele intermediare. 
	Pentru ca testarea ie destul de mult, am lasat la rulat checkerul meu si am pus rezultatul in
My_Checker_Output, pentru a va fi mai usor la corectat.
	Mentionez ca programul rulat pe masina locala FUNCTIONEAZA CORECT, dupa cum se poate observa si
din outputul checkerului.
	Dimensiunile pozelor sunt egale(verificarea e facuta cu diff).
	Rezultatul este corect chiar daca se aplica mai multe filtre.
	Makefile-ul are regulile build si clean si functioneaza cum trebuie.
	
~~~~~~~~~~~~~~~~~~~~
	   SCALARE
~~~~~~~~~~~~~~~~~~~~

	Pentru a arata cat mai bine ca programul scaleaza, am repetat aplicarea filtrului de 9 ori.
	Scalarea o voi documenta folosind cele mai mari 2 poze, pe care aplic bssembssem:
	
	PGM- Rorschach
	4 procese: 26.873s
	3 procese: 28.513s
	2 procese: 36.036s
	1 proces:  57.695s
	
	PNM- Landscape
	4 procese: 1m18.781s
	3 procese: 1m25.503s
	2 procese: 1m56.193s
	1 proces:  3m8.557s
	
	Programul a fost rulat pe un Intel Core m7-6y75, un procesor dual core cu hyper-threading
(2 cores, 4 threads). De aceea intre 1 si 2 core-uri se observa o scadere semnificativa a
timpului. Dupa aceea, scaderea este mai putin semnificativa. 

	Cum scaleaza programul?
	Initial fiecare proces citeste imaginea, sub forma unei zone continue de memorie, asa 
cum am spus si mai sus. Fiecare proces are o parte de imagine, calculata impartind vectorul
care tine imaginea in parti egale, fix ca in cazul laboratorului de la inceputul semestrului.
Acestea aplica filtrul corespunzator pe partea lor de imagine. Astfel, munca este impartita 
facil intre procese, care paralelizeaza calculele, ducand la scaderea timpului de rulare. 

PS: In urma une interventii a responsabilului pe forum, s-a precizat ca un singur proces trebuie
sa citeasca poza si s-o trimita tuturor celorlalte. Am modificat(minimal) tema cat sa satisfaca
aceasta cerinta comunicata ulterior, fara sa afecteze timpii de rulare.
	
PS2: Script de checker modificat. Inainte nu mergea bine daca folderul dat ca prim argument nu 
continea direct folderele pgm/pnm, care imi spun pe ce tip de imagine urmeaza sa rulez programul.
Acum ar trebui sa fie ok.
