#include <iostream>
#include <cmath>

using namespace std;


void sayHi(string name, int age);  // 函数在main()后面，前面要加这句话

double cube(double num){
    double result = num * num * num;
    return result;
}

int getMax(int num1, int num2, int num3){
    int result;

//    if(num1 > num2){
//        result = num1;
//    } else {
//        result = num2;
//    }
    if(num1 >= num2 && num1 >= num3){
        result = num1;
    } else if(num2 >= num1 && num2 >= num3){
        result = num2;
    } else {
        result = num3;
    }

    return result;
}

string getDayOfWeek(int dayNum){
    string dayName;
    switch(dayNum){
    case 0:
        dayName = "Sunday";
        break;
    case 1:
        dayName = "Monday";
        break;
    case 2:
        dayName = "Tuesday";
        break;
    case 3:
        dayName = "Wednesday";
        break;
    case 4:
        dayName = "Thursday";
        break;
    case 5:
        dayName = "Friday";
        break;
    case 6:
        dayName = "Saturday";
        break;
    default:
        dayName = "Invalid Day Number";
    }
    return dayName;
}

int power(int baseNum, int powNum){
    int result = 1;
    for(int i = 0; i < powNum; i++){
        result = result * baseNum;
    }
    return result;
}

class Book {
        public:
            string title;
            string author;
            int pages;
            Book(){
                title = "no title";
                author = "no author";
                pages = 0;
            }
            Book(string aTitle, string aAuthor, int aPages){
                // cout << aTitle << endl;
                title = aTitle;
                author = aAuthor;
                pages = aPages;
            }
};

class Student {
    public:
        string name;
        string major;
        double gpa;
        Student(string aName, string aMajor, double aGpa){
            name = aName;
            major = aMajor;
            gpa = aGpa;
        }

        bool hasHonors(){
            if(gpa >= 3.5){
                return true;
            }
            return false;
        }
};

class Movie{
    private:
        string rating;
    public:
        string title;
        string director;
        // string rating;
        Movie(string aTitle, string aDirector, string aRating){
            title = aTitle;
            director = aDirector;
            // rating = aRating;
            setRating(aRating);
        }
        void setRating(string aRating){
            // rating = aRating;
            if(aRating == "G" || aRating == "PG" || aRating == "PG-13" || aRating == "R" || aRating == "NR"){
                rating = aRating;
            } else {
                rating = "NR";
            }
        }

        string getRating(){

            return rating;
        }
};

class Chef{
    public:
        void makeChicken(){
            cout << "The chef makes yummy chicken" << endl;
        }
        void makeSalad(){
            cout << "The chef makes salad" << endl;
        }
        void makeSpecialDish(){
            cout << "The chef makes bbq ribs" << endl;
        }
};

class ItalianChef : public Chef{
    public:
        void makePasta(){
            cout << "The chef makes pasta" << endl;
        }
        void makeSpecialDish(){
            cout << "The chef makes chicken parm" << endl;
        }
};

int main()
{
//    cout << "   /|" << endl;
//    cout << "  / |" << endl;
//    cout << " /  |" << endl;
//    cout << "/___|" << endl;

//    string Name = "John";
////    int Age = 30;
//    int Age;
//    Age = 30;
//    cout << "There once was a man named " << Name << endl;
//    cout << "He was " << Age << " years old" << endl;

//    string phrase = "hello world";
//    cout << phrase.length() << endl;
//    cout << phrase[0] << endl;
//    cout << phrase.find("world", 0) << endl;
//    cout << phrase.substr(3, 3);

//    int wnum = 5;
//    double dnum = 5.5;
//    wnum++;
//    cout << wnum << endl;
//    cout << 5 + 5.5 << endl;
//    cout << 10 / 3 << endl;
//    cout << 10.0 / 3.0 << endl;

//    // cmath
//    cout << pow(2, 5) << endl;
//    cout << sqrt(36) << endl;
//    cout << round(4.3) << endl;
//    cout << ceil(4.1) << endl;
//    cout << floor(4.8) << endl;
//    cout << fmax(3, 10) << endl;
//    cout << fmin(1, 10) << endl;

//    int age;
//    cout << "Enter your age: ";
//    cin >> age;
//    cout << "You are " << age << " years old";
//
//    string name;
//    cout << "Enter your name: ";
//    getline(cin, name);
//    cout << "Hello " << name;

//    int luckyNums[] = {1, 2, 3, 4, 5, 6, 7};\
//    cout << luckyNums[0];

//    sayHi("Kevin", 28);
//    sayHi("Kevin", 28);
//    sayHi("Kevin", 28);

//    double answer = cube(5.0);
//    cout << answer;

//    bool isMale = false;
//    bool isTall = false;
//
//    if(isMale && isTall){
//        cout << "You are a tall male";
//    }
//    else if(isMale && !isTall){
//
//        cout << "You are a short male";
//    }
//    else if(!isMale && isTall){
//        cout << "You are tall but not male";
//    }
//    else{
//        cout << "You are not male and not tall";
//    }

//    cout << getMax(2, 50, 10);

//    // Building a Better Calculator
//    int num1, num2;
//    char op;
//    cout << "Enter first number: ";
//    cin >> num1;
//    cout << "Enter operator: ";
//    cin >> op;
//    cout << "Enter second number: ";
//    cin >> num2;
//    int result;
//    if(op == '+'){
//        result = num1 + num2;
//    } else if(op == '-'){
//        result = num1 - num2;
//    } else if(op == '/'){
//        result = num1 / num2;
//    } else if(op == '*'){
//        result = num1 * num2;
//    } else {
//        cout << "Invalid Operator";
//    }
//    cout << result;

//    // Switch Statements
//    cout << getDayOfWeek(0);

//    // While Loops
//    int index = 1;
//    while(index <= 5){
//        cout << index << endl;
//        index++;
//    }

//    // For Loops
//    for(int i = 1; i <= 5; i++){
//        cout << i << endl;
//    }

//    // Exponent Function
//    cout << power(2, 3);

//    // 2d Arrays & Nested Loops，嵌套循环
//    int numberGrid[3][2] = {
//                            {1, 2},
//                            {3, 4},
//                            {5, 6}
//                        };
//    cout << numberGrid[0][1] << endl;
//    for(int i = 0; i < 3; i++){
//        for(int j = 0; j < 2; j++){
//            cout << numberGrid[i][j];
//        }
//        cout << endl;
//    }

//    // Pointers
//    int age = 19;
//    int *pAge = &age;
//    double gpa = 2.7;
//    double *pGpa = &gpa;
//    string name = "Kevin";
//    string *pName = &name;
//    cout << &age << endl;
////    cout << "Age: " << &age << endl;
////    cout << "Gpa: " << &gpa << endl;
////    cout << "Name: " << &name << endl;
//    cout << pAge << endl;
//    cout << *pAge;

    // Classes & Objects

//    // Constructor Functions
//    // Book book1;
//    Book book1("Harry Potter", "JK Rowling", 500);
//    // Book book2;
//    Book book2("Lord of the Rings", "Tolkein", 700);
//    Book book3;
//    cout << book1.title << endl;
//    cout << book3.title;

//    // Object Functions
//    Student student1("Jim", "Business", 2.4);
//    Student student2("Pam", "Art", 3.6);
//    cout << student1.hasHonors();
//    cout << student2.hasHonors();

//    // Getters & Setters
//    Movie avengers("The Avengers", "Joss Whedon", "PG-13");
//    avengers.setRating("Dog");
//    // cout << avengers.rating;
//    cout << avengers.getRating();

    // Inheritance
    Chef chef;
    // chef.makeChicken();
    chef.makeSpecialDish();
    ItalianChef italianChef;
    // italianChef.makeChicken();
    // italianChef.makePasta();
    italianChef.makeSpecialDish();

    return 0;
}


void sayHi(string name, int age){
    cout << "Hello " << name << " you are " << age << endl;
}
