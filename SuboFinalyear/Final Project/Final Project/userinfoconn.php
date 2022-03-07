<?php
$host = "localhost";
$user = "root";
$pass = "";
$dbname = "userinfo";
$conn = mysqli_connect($host, $user, $pass, $dbname);
if($conn){

}

if(isset($_POST['submit']))
{
    $name = $_POST['name'];
    $age = $_POST['age'];
    $weight_in_kgs = $_POST['weight_in_kgs'];
    $height_in_cms = $_POST['height_in_cms'];
    $what_do_you_want_to_do = $_POST['what_do_you_want_to_do'];
    $how_much_is_your_daily_acitivity = $_POST['how_much_is_your_daily_acitivity'];
    $gender = $_POST['gender'];
    

    $query = "INSERT into User_Info(name, age, weight_in_kgs, height_in_cms, what_do_you_want_to_do, how_much_is_your_daily_acitivity, gender) values('$name', '$age', '$weight_in_kgs', '$height_in_cms', '$what_do_you_want_to_do', '$how_much_is_your_daily_acitivity', '$gender')";

    $result = mysqli_query($conn, $query);


}
// if($result){
//     header('location:welcome.php');
// }
